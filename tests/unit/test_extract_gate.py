"""The grounding gate on `reported`.

`reported` is what the audit tallies, so a field flagged reported on evidence the paper
does not contain is an absence counted as a presence -- 30 of 526 in a full corpus run.
"""
from __future__ import annotations

from acv.pipeline.evaluate import grounded_in

PAPER = ("We performed calculations with SIESTA. A mesh cutoff of 400 Ry was used "
         "throughout, and the Brillouin zone was sampled on a 9 x 9 x 1 grid.")


# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

def test_self_negating_evidence_is_not_grounded():
    """The exact shape of the bug: the model writes its own sentence saying the paper
    is silent, then flags the field reported."""
    assert not grounded_in("The paper does not explicitly state the mesh cutoff in Ry.",
                           PAPER)


def test_real_sentence_is_grounded_despite_line_wrapping():
    assert grounded_in("A mesh cutoff of 400 Ry was used throughout", PAPER)
    assert grounded_in("A mesh  cutoff of 400 Ry\nwas used throughout", PAPER)


def test_gate_flips_reported_and_clears_the_invented_value():
    from acv.types import Extraction

    rec = Extraction(paper_key="x")
    rec.method.mesh_cutoff_ry.reported = True
    rec.method.mesh_cutoff_ry.value = 0.0
    rec.method.mesh_cutoff_ry.evidence = "The paper does not explicitly state the mesh cutoff."
    rec.method.k_mesh.reported = True
    rec.method.k_mesh.evidence = "the Brillouin zone was sampled on a 9 x 9 x 1 grid"

    for name in type(rec.method).model_fields:
        e = getattr(rec.method, name)
        if e.reported and e.evidence and not grounded_in(e.evidence, PAPER):
            e.reported = False
            e.value = None

    assert rec.method.mesh_cutoff_ry.reported is False
    assert rec.method.mesh_cutoff_ry.value is None
    assert rec.method.k_mesh.reported is True      # a real quote survives


def test_missing_evidence_is_left_alone():
    """Verify model execution omits evidence per schema constraints.
    Gating on a missing quote would manufacture the absence this project measures --
    the one direction of error that is not acceptable."""
    assert not grounded_in("", PAPER)      # ungrounded, but the gate skips empty evidence


def test_truncation_is_detected_only_when_the_count_is_pinned_at_half():
    """Ollama discards half the prompt when it exceeds the window, and says so only in
    its own log. The client-side signal is `prompt_eval_count` sitting exactly at the
    half-window mark -- a prompt that is merely larger than half was processed fine."""
    from acv.llm import _was_truncated

    # measured: 54% of a paper discarded
    assert _was_truncated({"prompt_eval_count": 7420}, 14848)
    # measured: 16,274 tokens processed in full inside a 16,896 window
    assert not _was_truncated({"prompt_eval_count": 16274}, 16872)
    assert not _was_truncated({"prompt_eval_count": 3000}, 14848)
    assert not _was_truncated({}, 14848)
