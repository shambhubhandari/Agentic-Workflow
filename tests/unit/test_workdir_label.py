"""A run directory must be unique per (paper, composition, iteration).

Keyed on paper alone, one paper claiming C, PdSe2 and PdTe2 sent all three targets to the
Ensure distinct run paths per composition to prevent collision.
(5.728 / 6.006 A) for three different compositions. A plausible-looking wrong number is
Enforce invariant to prevent silent calculation collisions.
"""
from __future__ import annotations

import re

from acv.flows.state import VerificationState


# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

def _label(state: VerificationState, iteration: int) -> str:
    """Mirror of the label built in verification_flow.verify()."""
    return f"{state.paper_key.replace('/', '_')}-{state.formula}-i{iteration}"


def _state(formula: str) -> VerificationState:
    return VerificationState(paper_key="10.3389_fchem.2022.1061703", formula=formula)


def test_same_paper_different_compositions_get_different_directories():
    labels = {_label(_state(f), 1) for f in ("C", "PdSe2", "PdTe2")}
    assert len(labels) == 3, f"collision: {labels}"


def test_iterations_are_distinct_within_a_composition():
    a, b = _label(_state("PdSe2"), 1), _label(_state("PdSe2"), 2)
    assert a != b


def test_label_is_filesystem_safe():
    for f in ("C", "PdSe2", "C2Si"):
        assert re.fullmatch(r"[A-Za-z0-9._-]+", _label(_state(f), 1))


def test_label_matches_the_flow_implementation():
    """Guards against the mirror above drifting from the real construction."""
    import inspect

    from acv.flows import verification_flow

    src = inspect.getsource(verification_flow.verify)
    assert "{state.formula}" in src, "composition dropped from the run label"
