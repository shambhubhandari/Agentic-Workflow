"""The evaluator must not punish the extractor for being right.

Robustly compare extracted values against textual claims.
conversions are pinned here.
"""
from __future__ import annotations

# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

import pytest

from acv.pipeline.evaluate import _is_grounded, _norm, _numbers_in, _supports


@pytest.mark.parametrize("value,sentence,via", [
    # Every one of these was measured coming out of the real extractor.
    (680.285, "A suitable energy cutoff of 50 Ry was considered", "Ry->eV"),
    (816.34, "an energy cutoff of 60 Ry", "Ry->eV"),
    (0.2721, "a confinement energy displacement of 0.02 Ry was applied", "Ry->eV"),
    (0.02571, "the force threshold was set to 0.001 Ry/bohr", "Ry/bohr->eV/A"),
    (20.0, "a supercell dimension of 2.0 nm along the vacuum axis", "nm->A"),
    (550.0, "The cutoff energy for the plane-wave basis was set to 550 eV.", "direct"),
])
def test_unit_conversions_count_as_supported(value, sentence, via):
    ok, found = _supports(value, sentence)
    assert ok, f"{value} should be derivable from {sentence!r}"
    assert via.split()[0] in found or found == "direct"


def test_scientific_notation_forms():
    """`10-5 eV` in a PDF means 1e-5, not 10 and -5."""
    ok, _ = _supports(1e-05, "The convergence accuracy for energy was 10-5 eV")
    assert ok


@pytest.mark.parametrize("value,sentence", [
    (3.64, "Spin-polarization was considered in the adsorption process"),
    (999.0, "a plane-wave cutoff of 550 eV was used"),
])
def test_absent_values_are_rejected(value, sentence):
    """The check must still fail when the number is not there -- otherwise it measures
    nothing."""
    ok, _ = _supports(value, sentence)
    assert not ok


def test_grounding_tolerates_pdf_line_wrapping_but_not_fabrication():
    paper = _norm("All calculations were performed with the SIESTA software suite. "
                  "A 20 A vacuum layer was inserted.")
    assert _is_grounded("All calculations were performed with the SIESTA software suite",
                        paper) == "exact"
    # the same sentence as a PDF would break it across a column
    assert _is_grounded("All  calculations   were performed with the SIESTA\nsoftware suite",
                        paper) == "exact"
    # a sentence the paper does not contain
    assert _is_grounded("We used the VASP code with PAW pseudopotentials", paper) == "absent"
    assert _is_grounded("", paper) == "no_evidence"


def test_numbers_in_handles_the_forms_papers_actually_use():
    found = _numbers_in("cutoff 550 eV, threshold 1.0e-4, mesh 1.2 x 10-3, spacing 20 A")
    for expected in (550.0, 1.0e-4, 20.0):
        assert any(abs(f - expected) < 1e-9 for f in found), f"missed {expected}"


def test_absence_is_never_checked():
    """A field the paper did not report has nothing to ground -- and absence is the
    finding, not an error to score."""
    from acv.pipeline.evaluate import evaluate_one
    from acv.types import Extraction

    rec = Extraction(paper_key="x")          # every field defaults to reported=False
    assert evaluate_one(rec, "some paper text") == []


def test_superscript_exponents_are_read_as_numbers():
    """Papers typeset thresholds as `10⁻⁶ eV`; PDF extraction keeps the superscripts."""
    ok, _ = _supports(1e-06, "the total energy converged to within 10⁻⁶ eV")
    assert ok
    ok, _ = _supports(0.01, "forces on each atom were below 10⁻² eV Å⁻¹")
    assert ok


def test_booleans_are_grounded_only():
    """`bool` is a subclass of `int`. Asking whether True appears as a number in the
    sentence scores every spin-polarised=True as unsupported, which buried the real
    signal under 35 spurious failures."""
    from acv.pipeline.evaluate import evaluate_one
    from acv.types import Extraction

    rec = Extraction(paper_key="x")
    rec.method.spin_polarised.reported = True
    rec.method.spin_polarised.value = True
    rec.method.spin_polarised.evidence = "Spin-polarization was considered throughout"
    (check,) = evaluate_one(rec, "Spin-polarization was considered throughout the study.")
    assert check.grounded is True
    assert check.supported is None      # not applicable, not failed


def test_self_negating_sentence_is_not_matched_by_letter_frequency():
    """`quick_ratio()` compares character counts and ignores order, so any ordinary
    English sentence scores highly against any English text. It rated a model-authored
    "the paper does not explicitly state..." at 0.867 against a paper that never contains
    it -- above the 0.85 floor -- which let self-negating evidence through the gate it
    exists to stop. The decision must use ratio(), which respects sequence."""
    paper = _norm("We used SIESTA with a DZP basis. The Brillouin zone was sampled on a "
                  "9 x 9 x 1 Monkhorst-Pack grid and the structures were relaxed until "
                  "forces fell below 0.01 eV/A.")
    assert _is_grounded("The paper does not explicitly state the mesh cutoff for NAOs",
                        paper) == "absent"
    # a real sentence, PDF-mangled, must still be found
    assert _is_grounded("The Brillouin  zone was sampled on a 9 x 9 x 1\nMonkhorst-Pack grid",
                        paper) in ("exact", "fuzzy")
