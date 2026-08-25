"""System: test_epistemic module.

Strict validation of bounds on what may be claimed.
"""

from __future__ import annotations

# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

import pytest

from acv.guardrails import epistemic
from acv.guardrails.epistemic import EpistemicViolation, Verdict


class TestAxisConvention:
    """Enforce invariance to in-plane axis naming conventions."""

    def test_sorted_comparison_survives_a_swapped_convention(self):
        # Sorted comparisons decouple results from axis-naming conventions.
        ours = (6.145, 6.437)
        theirs = (6.437, 6.145)        # they call the longer one a
        verdict, _ = epistemic.decide_cell(
            ours, theirs, calibrated_offset=None, converged=True
        )
        assert verdict is Verdict.CONSISTENT

    def test_a_genuinely_different_cell_is_still_divergent(self):
        verdict, _ = epistemic.decide_cell(
            (6.145, 6.437), (5.500, 5.700), calibrated_offset=None, converged=True
        )
        assert verdict is Verdict.DIVERGENT

    def test_cell_reproduces_only_if_both_axes_do(self):
        # Short axis agrees, long axis does not.
        verdict, _ = epistemic.decide_cell(
            (6.145, 6.437), (6.145, 7.500), calibrated_offset=None, converged=True
        )
        assert verdict is Verdict.DIVERGENT


class TestVerdicts:
    def test_unconverged_is_inconclusive_however_close_the_numbers(self):
        verdict, why = epistemic.decide(
            3.640, 3.640, calibrated_offset=None, converged=False
        )
        assert verdict is Verdict.INCONCLUSIVE
        assert "not converged" in why

    def test_missing_value_is_not_attempted_not_divergent(self):
        verdict, _ = epistemic.decide(
            None, 3.640, calibrated_offset=None, converged=True
        )
        assert verdict is Verdict.NOT_ATTEMPTED

    def test_threshold_is_always_named_in_the_rationale(self):
        # Enforce threshold visibility in the generated rationale.
        _, why = epistemic.decide(3.700, 3.640, calibrated_offset=None, converged=True)
        assert f"{100 * epistemic.TOLERANCE:.2f}%" in why

    def test_calibrated_offset_overrides_the_default_tolerance(self):
        # Verify calibration overrides intercept default threshold behaviors.
        args = dict(converged=True)
        assert epistemic.decide(3.6946, 3.640, calibrated_offset=None, **args)[0] \
            is Verdict.DIVERGENT
        assert epistemic.decide(3.6946, 3.640, calibrated_offset=0.02, **args)[0] \
            is Verdict.CONSISTENT

    def test_there_is_no_verdict_meaning_verified_or_refuted(self):
        values = {v.value for v in Verdict}
        assert values == {"consistent", "divergent", "inconclusive", "not_attempted"}


class TestForbiddenText:
    @pytest.mark.parametrize(
        "text",
        [
            "the paper is wrong",
            "this refutes their result",
            "the study fails to reproduce",
            "an incorrect value",
            "these results cannot be trusted",
        ],
    )
    def test_assertions_of_fault_are_refused(self, text):
        with pytest.raises(EpistemicViolation):
            epistemic.check_text(text)

    def test_the_permitted_phrasing_passes(self):
        epistemic.check_text(
            "does not reproduce under an independent implementation"
        )

    def test_every_rationale_the_guardrail_itself_emits_is_permitted(self):
        # Ensure generated rationales comply with lexicon constraints.
        for ours, theirs, converged in [
            (3.640, 3.640, True), (3.900, 3.640, True), (3.640, 3.640, False)
        ]:
            _, why = epistemic.decide(
                ours, theirs, calibrated_offset=None, converged=converged
            )
            epistemic.check_text(why)
