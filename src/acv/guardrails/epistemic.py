"""System: epistemic module.

Provides strict, deterministic logic and strict typing for epistemic operations.
"""
from __future__ import annotations

# =============================================================================
#                     ********* SAFETY GUARDRAILS *********                    
#                       Strict definitions for epistemic.                      
# =============================================================================

import re
from enum import Enum
from typing import Optional



from ..settings import effective as _eff          # noqa: E402

TOLERANCE = float(_eff('tier2', 'tolerance', None, 0.01))


class Verdict(str, Enum):
    CONSISTENT = "consistent"
    """Agrees with the paper within the calibrated offset for this property."""

    DIVERGENT = "divergent"
    """Differs by more than the calibrated offset. This is a statement about our
    calculation and theirs jointly, never about the paper alone."""

    INCONCLUSIVE = "inconclusive"
    """Cannot distinguish. The default, and the honest answer whenever calibration is
    missing, convergence is unproven, or the difference sits near the threshold."""

    NOT_ATTEMPTED = "not_attempted"
    """No calculation was run."""


# Strict lexicon gate for forbidding subjective phrasing in generated reports.
FORBIDDEN_PATTERNS = [
    r"\bthe (paper|study|authors?|work) (is|are|was|were) (wrong|incorrect|mistaken)\b",
    r"\b(wrong|incorrect|erroneous|false|bogus|fabricated)\s+(value|result|number|data)\b",
    r"\bpaper('s)?\s+(error|mistake)\b",
    r"\b(refut\w+|disprov\w+|debunk\w+)\b",
    r"\bfail(s|ed)? to reproduce\b",          # implies fault; use "does not reproduce"
    r"\bcannot be trusted\b",
    r"\b(sloppy|careless|negligent)\b",
]

_COMPILED = [re.compile(p, re.I) for p in FORBIDDEN_PATTERNS]


class EpistemicViolation(RuntimeError):
    """Text made a claim about fault that the evidence cannot support."""


def check_text(text: str, *, strict: bool = True) -> list[str]:
    """Return the forbidden phrasings found in `text`.

    With strict=True (the default) a match raises. Report-writing paths should call this
    on anything destined for a human reader or a publication.
    """
    hits = [m.group(0) for pattern in _COMPILED for m in pattern.finditer(text or "")]
    if hits and strict:
        raise EpistemicViolation(
            "text asserts fault, which the evidence cannot support: "
            + ", ".join(repr(h) for h in hits)
            + ". Use 'does not reproduce under an independent implementation'."
        )
    return hits


def decide_cell(
    ours: tuple[float, float],
    theirs: tuple[float, float],
    *,
    calibrated_offset: Optional[float],
    converged: bool,
) -> tuple[Verdict, str]:
    """Compare two in-plane cells without trusting the axis LABELS.

    Papers do not agree on which axis is "a". One reports PdTe2 as a=6.437, b=6.145
    Sorting ascending standardizes the comparison axis.
    label-to-label produced a spurious -5.2% where the axis-matched agreement is -0.7%.

    Sorting both pairs before comparing removes the convention entirely, and is valid
    because an in-plane cell is unordered: {a, b} is the same cell as {b, a}.
    """
    ours_sorted, theirs_sorted = sorted(ours), sorted(theirs)
    verdicts, notes = [], []
    for label, o, t in zip(("short axis", "long axis"), ours_sorted, theirs_sorted):
        v, why = decide(o, t, calibrated_offset=calibrated_offset, converged=converged)
        verdicts.append(v)
        notes.append(f"{label}: {why}")

    # The cell reproduces only if BOTH axes do.
    if any(v is Verdict.INCONCLUSIVE for v in verdicts):
        return Verdict.INCONCLUSIVE, "; ".join(notes)
    if all(v is Verdict.CONSISTENT for v in verdicts):
        return Verdict.CONSISTENT, "; ".join(notes)
    return Verdict.DIVERGENT, "; ".join(notes)


def decide(
    ours: Optional[float],
    theirs: Optional[float],
    *,
    calibrated_offset: Optional[float],
    converged: bool,
    relative: bool = True,
) -> tuple[Verdict, str]:
    """Turn two numbers into a verdict, refusing to overclaim.

    The reported quantity is the DIFFERENCE between the published value and our SIESTA
    recomputation. It is not decomposed into "the paper is wrong" and "the basis differs",
    and it does not need to be: the deviation is a measurement, and the distribution of
    deviations across the corpus is the result. Attributing each one to a cause would be a
    cross-code study, which this is not.

    `calibrated_offset` therefore refines the threshold when it exists and is optional.
    Absent it, TOLERANCE is used and named in the rationale, so no verdict ever rests on a
    number the reader cannot see.

    What stays forbidden is the leap from DIVERGENT to a claim about the authors. DIVERGENT
    means our recomputation did not land on their number; `check_text` and the Verdict
    vocabulary keep it there.
    """
    if ours is None or theirs is None:
        return Verdict.NOT_ATTEMPTED, "no value to compare"

    difference = abs(ours - theirs)
    if relative and theirs:
        difference = difference / abs(theirs)
    shown = f"{100 * difference:.2f}%" if relative else f"{difference:.4f}"

    if not converged:
        return (
            Verdict.INCONCLUSIVE,
            f"difference {shown}, but our calculation is not converged; "
            "no comparison is meaningful",
        )

    threshold = calibrated_offset if calibrated_offset is not None else TOLERANCE
    source = (
        f"calibrated offset ({100 * threshold:.2f}%)" if calibrated_offset is not None
        else f"stated tolerance ({100 * TOLERANCE:.2f}%)"
    )

    if difference <= threshold:
        return (
            Verdict.CONSISTENT,
            f"difference {shown} is within the {source}",
        )

    return (
        Verdict.DIVERGENT,
        f"difference {shown} exceeds the {source}; our SIESTA recomputation does not "
        "reproduce the published value",
    )
