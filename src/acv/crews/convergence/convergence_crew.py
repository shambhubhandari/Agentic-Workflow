"""Tier 2: convergence_crew module.

Provides strict, deterministic logic and strict typing for convergence_crew operations.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from ... import llm
from ...flows.state import ConvergenceStep, SiestaParameters
from ...hooks import audit_log

log = logging.getLogger(__name__)

from ...settings import effective as _eff          # noqa: E402

# =============================================================================
#                    ********* CONSTANTS & THRESHOLDS *********                  
#         Budget limits, convergence bounds, and parameter escalation ladders.     
# =============================================================================

MAX_ITERATIONS = int(_eff('tier2', 'convergence.max_iterations', None, 6))
BUDGET_SECONDS = int(_eff('tier2', 'convergence.budget_seconds', None, 14400))

# Convergence thresholds (per-atom).
ENERGY_TOL_EV_PER_ATOM = float(_eff('tier2', 'convergence.energy_tol_ev_per_atom', None, 0.005))
CELL_TOL_PCT = float(_eff('tier2', 'convergence.cell_tol_pct', None, 0.5))

# Ladders. The agent picks which to step; it cannot invent values off these.
MESH_LADDER = [float(x) for x in _eff('tier2', 'convergence.mesh_ladder', None, [200,300,400,500,700,900])]
BASIS_LADDER = list(_eff("tier2", "convergence.basis_ladder", None, ["SZ","SZP","DZ","DZP","TZP"]))


# =============================================================================
#                      ********* DECISION SCHEMAS *********                    
#          Strict Pydantic outputs for the convergence strategy agent.         
# =============================================================================

class Decision(BaseModel):
    """The agent's proposal for the next iteration."""

    action: str = Field(description="one of: increase_mesh, improve_basis, denser_k, stop")
    reasoning: str = ""
    expectation: str = Field(
        default="",
        description="What the agent expects to happen, so a wrong prediction is visible.",
    )


# =============================================================================
#                 ********* EVALUATION & ESCALATION *********                
#       Arithmetic tolerance checks and strictly upward parameter stepping.    
# =============================================================================

def is_converged(steps: list[ConvergenceStep]) -> tuple[bool, str]:
    """Arithmetic, not judgement.

    Requires two successive completed steps agreeing in BOTH energy per atom and cell
    parameters. A single run is never converged, however good it looks.
    """
    usable = [s for s in steps if s.energy_ev is not None and s.converged_scf]
    if len(usable) < 2:
        return False, f"only {len(usable)} completed calculation(s); need at least 2"

    # The deltas are computed by the flow against the previous completed step, so
    # only the last one is read here; the length check above is what guarantees a
    # previous step existed to compare against.
    last = usable[-1]
    d_energy = last.delta_energy_ev
    d_cell = last.delta_cell_pct

    if d_energy is None or d_cell is None:
        return False, "deltas not computed"
    if abs(d_energy) > ENERGY_TOL_EV_PER_ATOM:
        return False, (
            f"energy still moving: {abs(d_energy)*1000:.1f} meV/atom > "
            f"{ENERGY_TOL_EV_PER_ATOM*1000:.0f} meV/atom"
        )
    if abs(d_cell) > CELL_TOL_PCT:
        return False, f"cell still moving: {abs(d_cell):.2f}% > {CELL_TOL_PCT}%"

    return True, (
        f"converged: dE = {abs(d_energy)*1000:.1f} meV/atom, "
        f"dcell = {abs(d_cell):.2f}% between the last two settings"
    )


def _apply(action: str, params: SiestaParameters) -> tuple[SiestaParameters, str]:
    """Turn an action into new parameters, stepping only upward along the ladders."""
    new = params.model_copy(deep=True)

    if action == "increase_mesh":
        higher = [m for m in MESH_LADDER if m > params.mesh_cutoff_ry]
        if not higher:
            return new, "mesh cutoff already at the top of the ladder"
        new.mesh_cutoff_ry = higher[0]
        return new, f"mesh cutoff {params.mesh_cutoff_ry} -> {new.mesh_cutoff_ry} Ry"

    if action == "improve_basis":
        if params.basis.upper() not in BASIS_LADDER:
            new.basis = "DZP"
            return new, f"basis {params.basis} not on the ladder; set to DZP"
        idx = BASIS_LADDER.index(params.basis.upper())
        if idx >= len(BASIS_LADDER) - 1:
            return new, "basis already at the top of the ladder"
        new.basis = BASIS_LADDER[idx + 1]
        return new, f"basis {params.basis} -> {new.basis}"

    if action == "denser_k":
        kx, ky, kz = params.kgrid
        new.kgrid = [kx + 2, ky + 2, 1]     # vacuum axis stays at 1, always
        return new, f"k-grid {params.kgrid} -> {new.kgrid}"

    return new, f"unrecognised action {action!r}; parameters unchanged"


def _history(steps: list[ConvergenceStep]) -> str:
    if not steps:
        return "(no calculations yet)"
    lines = []
    for s in steps:
        energy = f"{s.energy_ev:.4f} eV" if s.energy_ev is not None else "failed"
        cell = ", ".join(f"{v:.3f}" for v in (s.cell or [])) or "-"
        deltas = ""
        if s.delta_energy_ev is not None:
            deltas = (f"  dE={s.delta_energy_ev*1000:+.1f} meV/atom"
                      f"  dcell={s.delta_cell_pct:+.2f}%")
        lines.append(
            f"  {s.iteration}. {s.parameters}  ->  {energy}  cell=[{cell}]{deltas}"
        )
    return "\n".join(lines)


# =============================================================================
#                      ********* AGENT DISPATCH *********                    
#       Orchestration of the LLM prompt, history formatting, and guards.       
# =============================================================================

def decide_next(
    steps: list[ConvergenceStep],
    params: SiestaParameters,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> tuple[Optional[SiestaParameters], Decision, str]:
    """Choose the next setting to try, or stop.

    Returns (next_parameters or None, decision, explanation). None means stop.
    """
    converged, why = is_converged(steps)
    if converged:
        return None, Decision(action="stop", reasoning=why), why

    if len(steps) >= MAX_ITERATIONS:
        return None, Decision(action="stop", reasoning="iteration cap"), (
            f"stopping after {len(steps)} iterations without convergence "
            f"(cap {MAX_ITERATIONS})"
        )

    spent = sum(s.seconds or 0 for s in steps)
    if spent > BUDGET_SECONDS:
        return None, Decision(action="stop", reasoning="budget"), (
            f"stopping: {spent/3600:.1f} h spent exceeds the "
            f"{BUDGET_SECONDS/3600:.0f} h budget for one claim"
        )

    if not steps:
        # Nothing to reason about yet; the translator's settings are the first point.
        return params, Decision(action="stop", reasoning="initial point"), "first calculation"

    prompt = f"""
You are deciding how to converge a SIESTA calculation of a 2D monolayer.

Current settings: basis={params.basis}, mesh_cutoff={params.mesh_cutoff_ry} Ry,
k-grid={params.kgrid}

History:
{_history(steps)}

Convergence requires two successive settings agreeing to better than
{ENERGY_TOL_EV_PER_ATOM*1000:.0f} meV/atom in energy AND {CELL_TOL_PCT}% in cell parameters.
Not yet met: {why}

Choose ONE action:
  increase_mesh  - step the real-space mesh cutoff up. Do this when the energy is still
                   moving but the cell is stable; mesh cutoff mainly affects the energy.
  improve_basis  - step the numerical-orbital basis up (SZ<SZP<DZ<DZP<TZP). Do this when
                   the CELL is still moving. Basis quality dominates geometry in an NAO
                   code, which is the usual cause of a wrong lattice parameter.
  denser_k       - increase in-plane k-sampling. Rarely the cause for an insulator with
                   an already reasonable grid.
  stop           - further refinement will not help.

State what you expect to happen, so a wrong prediction is visible afterwards.
"""

    _t0 = time.time()
    decision, model_used = llm.structured(
        prompt, Decision, agent="convergence", provider=provider, model=model
    )

    if decision.action == "stop":
        return None, decision, f"agent chose to stop: {decision.reasoning[:120]}"

    next_params, note = _apply(decision.action, params)
    _overrides: list[str] = []

    # Enforce strictly upward escalation.
    if (
        next_params.mesh_cutoff_ry < params.mesh_cutoff_ry
        or BASIS_LADDER.index(next_params.basis.upper()) < BASIS_LADDER.index(params.basis.upper())
        or sum(next_params.kgrid) < sum(params.kgrid)
    ):
        _overrides.append(f"refused: {decision.action} would make the run cheaper")
        audit_log.record("convergence", "convergence/config", model=model_used,
                         response=decision.model_dump(), proposed=decision.model_dump(),
                         applied=None, overrides=_overrides, seconds=time.time()-_t0,
                         context={"iteration": len(steps), "action": decision.action})
        log.warning("agent proposed a cheaper setting; refused")
        return None, decision, "refused: proposed setting is cheaper, not more converged"

    if next_params.model_dump() == params.model_dump():
        return None, decision, f"no further refinement available ({note})"

    audit_log.record(
        "convergence", "convergence/config", model=model_used,
        prompt_tail=_history(steps), response=decision.model_dump(),
        proposed=decision.model_dump(), applied=next_params.model_dump(),
        overrides=_overrides, seconds=time.time() - _t0,
        context={"iteration": len(steps), "action": decision.action, "note": note[:60]},
    )
    return next_params, decision, note
