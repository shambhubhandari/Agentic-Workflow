"""Calibration sufficiency is a precondition, not a check an agent may elect to run.

It was previously enforced only when the Critic happened to call
`calibration_for_structure`. Two papers on the same composition, in the same campaign, got
`divergent` and `inconclusive` respectively -- one invocation looked, the other did not.
Whether the reference set holds enough distinct structures is arithmetic, so it must be
settled in code before the Critic is consulted.
"""
from __future__ import annotations

import json

from acv.flows import verification_flow as vf


# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

def test_shortfall_reported_when_too_few_structures(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"usable": False, "distinct_structures": ["CN2", "PdSe2"]}))
    monkeypatch.setattr(vf, "PROCESSED", tmp_path)
    msg = vf._calibration_shortfall()
    assert msg and "2 distinct" in msg and "6 required" in msg


def test_no_shortfall_when_calibration_is_usable(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "usable": True,
        "distinct_structures": ["CN2", "PdSe2", "C", "Si", "BN2", "PdTe2"]}))
    monkeypatch.setattr(vf, "PROCESSED", tmp_path)
    assert vf._calibration_shortfall() is None


def test_missing_calibration_is_a_shortfall_not_a_pass(tmp_path, monkeypatch):
    """Absent calibration must never read as 'calibrated'."""
    monkeypatch.setattr(vf, "PROCESSED", tmp_path)
    assert vf._calibration_shortfall() == "no calibration has been run"


def test_unreadable_calibration_is_a_shortfall(tmp_path, monkeypatch):
    (tmp_path / "calibration.json").write_text("{not json")
    monkeypatch.setattr(vf, "PROCESSED", tmp_path)
    assert "unreadable" in (vf._calibration_shortfall() or "")


def test_gate_is_applied_before_the_critic_in_verify():
    """The precondition must not sit inside _apply_critic."""
    import inspect
    src = inspect.getsource(vf.verify)
    gate = src.index("_calibration_shortfall()")
    critic = src.index("_apply_critic(state)")
    assert gate < critic, "calibration gate must precede the Critic review"
