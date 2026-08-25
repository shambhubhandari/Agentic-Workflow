"""Tier 2: state module.

Provides strict, deterministic logic and strict typing for state operations.
"""
from __future__ import annotations

# =============================================================================
#                    ********* ORCHESTRATION FLOWS *********                   
#                         Strict definitions for state.                        
# =============================================================================

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..guardrails.epistemic import Verdict


class Stage(str, Enum):
    START = "start"
    TRANSLATED = "translated"
    CONVERGING = "converging"
    CONVERGED = "converged"
    DIAGNOSING = "diagnosing"
    COMPARED = "compared"
    ABANDONED = "abandoned"


class Derivation(BaseModel):
    """Which paper parameter justified one SIESTA setting."""

    setting: str
    justification: str


class SiestaParameters(BaseModel):
    """SIESTA settings, as decided by the Translator agent.

    Every field the agent could not derive from the paper is recorded in `unmapped`
    rather than filled with a default and forgotten. A parameter silently invented here
    would propagate into a verdict about somebody's published work.
    """

    # Field order dictates autoregressive generation sequence. Reasoning must precede decisions.
    reasoning: str = Field(
        default="",
        description="Your working: what the paper states, which SIESTA directives "
                    "correspond, and what has no equivalent. Written BEFORE the "
                    "settings so the settings follow from it.",
    )
    unmapped: list[str] = Field(
        default_factory=list,
        max_length=25,
        description=(
            "One COMPLETE SENTENCE per entry -- never a bare parameter name. Each "
            "sentence must name the parameter, give its value from the paper where one "
            "was stated, and say why it could not be carried over. Example: \"the "
            "paper's plane-wave cutoff (500 eV) has no NAO equivalent, so a mesh cutoff "
            "of 300 Ry was chosen independently\". "
            "Include a parameter here only if (a) the paper states it and SIESTA cannot "
            "reproduce it, or (b) SIESTA needs it and the paper never states it. Do NOT "
            "list a parameter the paper reported and SIESTA supports -- that is a "
            "successful mapping and belongs in derived_from instead. "
            "Never silently defaulted."
        ),
    )
    # Constrained decoding requires typed lists over arbitrary dict mappings.
    derived_from: list[Derivation] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "One entry for EACH of the five settings you decide below -- basis, "
            "mesh_cutoff_ry, kgrid, xc and xc_authors -- so five entries in total. "
            "State which paper parameter justified it, or say explicitly that the paper "
            "did not state one and give the standard value you used instead. A setting "
            "with no entry here is an unjustified setting."
        ),
    )

    # Omit defaults to force explicit model generation for target fields.
    basis: str
    mesh_cutoff_ry: float
    # Lists compile safely in JSON-Schema-to-grammar converters.
    kgrid: list[int] = Field(min_length=3, max_length=3)
    xc: str
    xc_authors: str


class ConvergenceStep(BaseModel):
    """One iteration of the convergence loop."""

    iteration: int
    parameters: dict[str, Any]
    energy_ev: Optional[float] = None
    cell: Optional[list[float]] = None
    converged_scf: bool = False
    delta_energy_ev: Optional[float] = None
    delta_cell_pct: Optional[float] = None
    seconds: Optional[float] = None
    decision: str = ""
    error: Optional[str] = None


class VerificationState(BaseModel):
    """Everything the Tier 2 flow knows about one claim under verification."""

    # inputs
    paper_key: str
    doi: Optional[str] = None
    formula: str = ""
    structure_path: Optional[str] = None
    claimed_property: str = ""
    claimed_value: Optional[float] = None
    claimed_unit: Optional[str] = None
    paper_method: dict[str, Any] = Field(default_factory=dict)

    # progress
    stage: Stage = Stage.START
    parameters: Optional[SiestaParameters] = None
    steps: list[ConvergenceStep] = Field(default_factory=list)

    # outcome
    our_value: Optional[float] = None
    # Persist intermediate axes to allow deterministic re-evaluation.
    our_a: Optional[float] = None
    our_b: Optional[float] = None
    verdict: Verdict = Verdict.NOT_ATTEMPTED
    rationale: str = ""
    failure_class: Optional[str] = None
    abandoned_reason: Optional[str] = None

    # Retain numeric evaluations regardless of the qualitative hold status.
    critic_verdict: Optional[str] = None
    critic_reasoning: str = ""
    critic_concerns: list[dict[str, Any]] = Field(default_factory=list)
    critic_tools_used: list[str] = Field(default_factory=list)
    provisional_verdict: Optional[Verdict] = None

    # provenance: which configuration produced this verdict
    config_fingerprint: Optional[str] = None
    tolerance_used: Optional[float] = None

    # accounting
    total_seconds: float = 0.0
    n_calculations: int = 0

    def last_step(self) -> Optional[ConvergenceStep]:
        return self.steps[-1] if self.steps else None
