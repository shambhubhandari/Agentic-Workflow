"""Tier 2: critic_crew module.

Provides strict, deterministic logic and strict typing for critic_crew operations.
"""
from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ... import llm
from ...hooks import audit_log
from ..loader import prompt as load_prompt

log = logging.getLogger(__name__)

from ...settings import effective as _eff          # noqa: E402

MAX_TOOL_ROUNDS = int(_eff('tier2', 'agents.critic_max_tool_rounds', None, 6))


# --------------------------------------------------------------------- bound checks
#
# The Critic chooses WHICH evidence to gather. It does not choose WHAT to gather it
# about -- that is a fact of the task, already sitting in VerificationState.
#
# Free-form tool calling asked the model to retype those facts into arguments, and a 4B
# model does not survive it: reviewing PdSe2 it called expected_symmetry for FeO, Cu and
# NaCl, none of which appear anywhere in this repository, then held the verdict because
# the fabricated lookups failed. Eight of the nine tools take arguments that are pure
# restatement of state, so the retyping bought nothing and cost a whole class of error.
#
# Binding them in code turns "emit a syntactically valid call with correct arguments"
# (hard) into "pick names from a fixed list" (a grammar can guarantee it). The agency is
# preserved where it is real -- the model still decides what is worth checking, and its
# choice is recorded -- and removed where it was only transcription.

CHECK_TOOLS: dict[str, str] = {
    "symmetry": "check_symmetry",
    "expected_symmetry": "expected_symmetry",
    "calibration": "calibration_for_structure",
    "convergence": "convergence_status",
    "units": "convert_units",
    "structure_file": "read_structure",
    "literature_claims": "literature_claims",
    "literature_methods": "literature_methods",
}

CheckName = Enum("CheckName", {k: k for k in CHECK_TOOLS}, type=str)


def _bind(check: str, state, prototype: str) -> Optional[dict[str, Any]]:
    """Arguments for a check, taken from state. None means the check is not runnable."""
    last = state.last_step() if hasattr(state, "last_step") else None
    cell = sorted((last.cell if last else None) or [])[:2]
    steps = [s for s in getattr(state, "steps", []) if s.energy_ev is not None]

    if check == "symmetry":
        if len(cell) < 2:
            return None
        return {"a": cell[0], "b": cell[1], "prototype": prototype}
    if check in ("expected_symmetry", "literature_claims", "literature_methods"):
        return {"reduced_formula": state.formula} if state.formula else None
    if check == "calibration":
        return {}
    if check == "convergence":
        if len(steps) < 2:
            return None
        return {
            "energies_ev": [s.energy_ev for s in steps],
            "cells_a": [sorted(s.cell)[0] for s in steps if s.cell],
            "n_atoms": max(getattr(state, "n_atoms", 0) or 6, 1),
        }
    if check == "units":
        # Enforce valid thickness requirements for unit conversions.
        unit = (state.claimed_unit or "").lower()
        if state.claimed_value is None or unit not in ("n/m", "gpa"):
            return None
        thickness = getattr(state, "thickness_ang", None)
        if not thickness:
            return None
        return {"value": state.claimed_value, "from_unit": state.claimed_unit,
                "thickness_ang": float(thickness)}
    if check == "structure_file":
        return {"path": state.structure_path} if state.structure_path else None
    return None


# =============================================================================
#                      ********* DECISION SCHEMAS *********                    
#          Strict Pydantic models bounding the generation grammar.             
# =============================================================================

class CheckPlan(BaseModel):
    """Which checks the Critic wants run. Names only."""
    checks: list[CheckName] = Field(
        default_factory=list,
        max_length=len(CHECK_TOOLS),
        description="The checks to run. Choose every one that could change the verdict.",
    )


def _run_checks(state, prototype: str, names: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Execute the selected checks with bound arguments. Registry still enforces access."""
    from ... import tools

    results: dict[str, Any] = {}
    used: list[str] = []
    for name in dict.fromkeys(names):                    # de-duplicate, keep order
        tool = CHECK_TOOLS.get(name)
        if tool is None:
            continue
        args = _bind(name, state, prototype)
        if args is None:
            results[name] = {"error": "not runnable: the state has no value to check"}
            continue
        try:
            results[name] = tools.call("critic", tool, args)
        except tools.ToolRefused as exc:
            results[name] = {"error": str(exc)}
        used.append(tool)
    return results, used


class Concern(BaseModel):
    """Strictly bounded concern schema."""

    category: str = Field(
        description="symmetry | convention | calibration | identity | residual | convergence"
    )
    severity: str = Field(description="blocking | warning")
    detail: str
    check: str = Field(description="the specific check a human should run")


class Review(BaseModel):
    """Strictly bounded review schema."""

    verdict: str = Field(description="pass or hold")
    reasoning: str = ""
    concerns: list[Concern] = Field(default_factory=list, max_length=6)
    tools_used: list[str] = Field(default_factory=list, max_length=12)


def _context(state, claim_value: Optional[float], prototype: str) -> str:
    last = state.last_step() if hasattr(state, "last_step") else None
    cell = (last.cell if last else None) or []
    history = [
        {
            "iteration": s.iteration,
            "parameters": s.parameters,
            "energy_ev": s.energy_ev,
            "cell": [round(x, 4) for x in (s.cell or [])][:2],
            "converged_scf": s.converged_scf,
        }
        for s in getattr(state, "steps", [])
    ]
    return json.dumps(
        {
            "material": state.formula,
            "prototype": prototype,
            "structure_path": state.structure_path,
            "paper_claim": {
                "property": state.claimed_property,
                "value": claim_value,
                "unit": state.claimed_unit,
            },
            "our_result": {
                "a": round(sorted(cell)[0], 4) if cell else None,
                "b": round(sorted(cell)[1], 4) if len(cell) > 1 else None,
            },
            "stage": getattr(state.stage, "value", str(state.stage)),
            "n_calculations": state.n_calculations,
            "history": history,
        },
        indent=2,
        default=str,
    )


# =============================================================================
#                      ********* AGENT DISPATCH *********                    
#         Two-phase orchestration: evidence gathering and judgement.           
# =============================================================================

def review(
    state,
    prototype: str = "tetragonal",
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Review:
    """Review one completed verification. Returns a Review; `hold` blocks the verdict."""
    instructions, prompt_id = load_prompt("critic", "critic", "review_verification")
    context = _context(state, state.claimed_value, prototype)
    started = time.time()

    # --- phase 1: choose the evidence -------------------------------------------
    # Enforce grammar-bound tool selection.
    plan, model_used = llm.structured(
        f"{instructions}\n\n---\n\n{context}\n\n"
        f"Available checks: {', '.join(CHECK_TOOLS)}.\n"
        "Name every check that could change this verdict. Arguments are supplied from "
        "the record above; you choose only what to check.",
        CheckPlan, agent="critic", provider=provider, model=model,
    )
    wanted = [c.value if hasattr(c, "value") else str(c) for c in plan.checks]

    # Fallback to full toolset if selection is empty.
    overrides: list[str] = []
    if not wanted:
        overrides.append("no checks selected; ran the full set")
        wanted = list(CHECK_TOOLS)

    evidence, used = _run_checks(state, prototype, wanted)

    # --- phase 2: judge, with the evidence in hand ------------------------------
    result, model_used = llm.structured(
        f"{instructions}\n\n---\n\n{context}\n\n"
        f"You asked for these checks: {', '.join(wanted)}\n"
        f"Their results:\n{json.dumps(evidence, indent=2, default=str)}\n\n"
        "Judge the verification against these results. Cite a result for every concern; "
        "a concern you cannot ground in one of them is not a concern.",
        Review, agent="critic", provider=provider, model=model,
    )
    parsed = result
    result.tools_used = used

    # Code gate: escalate blocking concerns to mandatory hold.
    if any(c.severity == "blocking" for c in result.concerns) and result.verdict != "hold":
        overrides.append("verdict forced to hold: a blocking concern was raised")
        result.verdict = "hold"
    if result.verdict not in ("pass", "hold"):
        overrides.append(f"unrecognised verdict {result.verdict!r}; defaulting to hold")
        result.verdict = "hold"

    audit_log.record(
        "critic", prompt_id, model=model_used,
        prompt_tail=context[-1200:], response=result.model_dump(),
        proposed={"verdict": parsed.verdict if parsed else None},
        applied={"verdict": result.verdict}, overrides=overrides,
        seconds=time.time() - started,
        context={"material": state.formula, "tools": ",".join(used) or "none"},
    )
    return result
