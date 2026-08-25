"""Tier 2: diagnostician_crew module.

Provides strict, deterministic logic and strict typing for diagnostician_crew operations.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ... import llm
from ...hooks import audit_log

log = logging.getLogger(__name__)

from ...settings import effective as _eff          # noqa: E402

MAX_RETRIES = int(_eff('tier2', 'agents.diagnostician_max_retries', None, 2))
TAIL_LINES = int(_eff('tier2', 'agents.diagnostician_tail_lines', None, 120))


class FailureClass(str, Enum):
    SCF_NOT_CONVERGED = "scf_not_converged"
    GEOMETRY_NOT_CONVERGED = "geometry_not_converged"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    INPUT_ERROR = "input_error"
    PSEUDOPOTENTIAL_ERROR = "pseudopotential_error"
    DISK_FULL = "disk_full"
    UNKNOWN = "unknown"


# Which classes are worth another attempt, and what to change. Retrying anything else
# just spends the budget twice on the same outcome.
RETRY_STRATEGY: dict[FailureClass, Optional[str]] = {
    FailureClass.SCF_NOT_CONVERGED: "reduce DM.MixingWeight and raise MaxSCFIterations",
    FailureClass.GEOMETRY_NOT_CONVERGED: "raise MD.NumCGsteps",
    FailureClass.TIMEOUT: "extend the wall clock, or reduce basis for a first pass",
    FailureClass.OUT_OF_MEMORY: "reduce basis size or MPI ranks",
    FailureClass.INPUT_ERROR: None,            # our bug; retrying changes nothing
    FailureClass.PSEUDOPOTENTIAL_ERROR: None,  # missing or broken file
    FailureClass.DISK_FULL: None,              # needs a human
    FailureClass.UNKNOWN: None,
}


# =============================================================================
#                      ********* DECISION SCHEMAS *********                    
#          Strict Pydantic model bounding the diagnostic response.             
# =============================================================================

class Diagnosis(BaseModel):
    failure_class: FailureClass
    evidence_line: str = Field(
        description="A line copied VERBATIM from the output that justifies the class."
    )
    reasoning: str = ""
    retry_recommended: bool = False


def _tail(text: str, n: int = TAIL_LINES) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines[-n:])


# =============================================================================
#                      ********* AGENT DISPATCH *********                    
#         Log truncation, prompt orchestration, and fallback logic.            
# =============================================================================

def diagnose(
    output_text: str,
    error: Optional[str] = None,
    attempt: int = 1,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Diagnosis:
    """Classify a failed SIESTA run. The citation is verified against the output."""
    tail = _tail(output_text)
    if not tail and not error:
        return Diagnosis(
            failure_class=FailureClass.UNKNOWN,
            evidence_line="",
            reasoning="no output and no error text to diagnose",
            retry_recommended=False,
        )

    prompt = f"""
A SIESTA calculation failed. Classify it.

Choose exactly one failure_class from:
  scf_not_converged       - the SCF cycle hit its iteration limit
  geometry_not_converged  - SCF fine, but the relaxation did not reach the force tolerance
  out_of_memory           - allocation failure, killed by the OOM reaper
  timeout                 - wall clock exceeded
  input_error             - malformed fdf, bad block, inconsistent species
  pseudopotential_error   - missing, unreadable or mismatched pseudopotential
  disk_full               - no space left
  unknown                 - none of the above fit

You MUST copy one line VERBATIM from the output below into evidence_line, as the
justification. Do not paraphrase it and do not compose a line. If nothing in the output
supports a class, choose unknown with an empty evidence_line.

Runner error: {error or "(none)"}

Output tail:
{tail}
"""
    _t0 = time.time()
    diagnosis, model_used = llm.structured(
        prompt, Diagnosis, agent="diagnostician", provider=provider, model=model
    )
    _proposed = diagnosis.model_dump()
    _overrides: list[str] = []

    # Verify that the diagnosis evidence exists in the raw text.
    cited = (diagnosis.evidence_line or "").strip()
    haystack = (output_text or "") + "\n" + (error or "")
    if cited and cited not in haystack:
        log.warning(
            "diagnostician cited a line absent from the output: %r", cited[:80]
        )
        diagnosis.reasoning = (
            f"[citation rejected: {cited[:60]!r} does not appear in the output] "
            + diagnosis.reasoning
        )
        _overrides.append("citation rejected: quoted line absent from the output")
        diagnosis.failure_class = FailureClass.UNKNOWN
        diagnosis.evidence_line = ""
        diagnosis.retry_recommended = False

    # Retry flags are evaluated strictly by taxonomy.
    strategy = RETRY_STRATEGY.get(diagnosis.failure_class)
    if strategy is None:
        diagnosis.retry_recommended = False
    elif attempt >= MAX_RETRIES:
        diagnosis.retry_recommended = False
        diagnosis.reasoning += f" [retry cap {MAX_RETRIES} reached]"
    else:
        diagnosis.retry_recommended = True

    audit_log.record(
        "diagnostician", "diagnostician/config", model=model_used,
        prompt_tail=tail[-800:], response=_proposed,
        proposed=_proposed, applied=diagnosis.model_dump(), overrides=_overrides,
        seconds=time.time() - _t0,
        context={"attempt": attempt, "class": diagnosis.failure_class.value},
    )
    return diagnosis
