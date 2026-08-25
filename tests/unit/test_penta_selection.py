"""Deciding what is, and is not, a pentagonal structure.

Selecting by chemical formula alone is the defect that ran through this project: it put
graphene, h-BN and hexagonal TMDs into a calibration set meant for penta materials, which
inflated the verdict threshold roughly threefold, and it matched a penta-PdSe2 claim
against the 1T polymorph of the same formula.
"""

from __future__ import annotations

# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

import pytest

from acv.pipeline import generate, penta
from acv.pipeline.calibrate import is_penta_prototype

CELL = [[5.7, 0, 0], [0, 5.9, 0], [0, 0, 20]]


def attrs(formula: str, heights: list[float]) -> dict:
    return {
        "chemical_formula_reduced": formula,
        "lattice_vectors": CELL,
        "cartesian_site_positions": [[0.0, 0.0, z] for z in heights],
    }


class TestPrototypeScreen:
    def test_accepts_the_penta_motif(self):
        # 2 M mid-plane, 2 X above, 2 X below.
        assert is_penta_prototype(attrs("PdSe2", [10, 10, 10.8, 10.8, 9.2, 9.2]))

    def test_rejects_the_hexagonal_polymorph_of_the_same_formula(self):
        # 1T/CdI2-type PdSe2. Its SIESTA relaxation sat 8.4% from its plane-wave
        # reference, which was then misread as a problem with the penta parent template.
        assert not is_penta_prototype(attrs("PdSe2", [10, 10.8, 9.2]))

    def test_rejects_a_2x1_tmd_supercell_that_mimics_the_motif(self):
        # A 2x1 cell of 2H-MoS2 has 6 sites, b/a = 1.000 and the same 2-up/2-mid/2-down
        # pattern. Geometry cannot separate it from penta; the composition check must.
        assert not is_penta_prototype(attrs("MoS2", [10, 10, 11.5, 11.5, 8.5, 8.5]))

    def test_rejects_a_planar_structure(self):
        assert not is_penta_prototype(
            attrs("NiP2", [10, 10, 10.02, 10.02, 9.98, 9.98])
        )

    def test_rejects_stacked_layers(self):
        # A 7.9 A span is several layers, not one.
        assert not is_penta_prototype(attrs("CN2", [10, 10, 13.9, 13.9, 6.1, 6.1]))

    def test_rejects_graphene(self):
        assert not is_penta_prototype(attrs("C", [10, 10]))


class TestSiteAssignment:
    @pytest.mark.parametrize(
        "formula,expected",
        [
            ("C", ("C", "C", None)),           # unitary: all six sites
            ("PdSe2", ("Pd", "Se", None)),     # binary: M:X = 1:2
            ("NiP2", ("Ni", "P", None)),
            ("C2Si", ("Si", "C", None)),       # penta-SiC2: silicon mid-plane
            ("BCP", ("C", "B", "P")),          # ternary: looked up, not derived
        ],
    )
    def test_sites_follow_from_the_prototype_stoichiometry(self, formula, expected):
        assert generate.sites_for(formula) == expected

    @pytest.mark.parametrize("formula", ["C3H2", "HSi"])
    def test_compositions_off_the_prototype_are_refused_not_guessed(self, formula):
        # 2 M + 4 X fixes M:X = 1:2. A composition that cannot sit on those sites must
        # fail loudly rather than enter a verdict as though it had been verified.
        assert generate.sites_for(formula) is None
        with pytest.raises(ValueError, match="penta prototype"):
            generate.from_formula(formula)


class TestTextDetection:
    def test_finds_a_penta_material(self):
        found, formula, _ = penta.detect("We study penta-graphene, a 2D carbon allotrope")
        assert found and formula == "C"

    def test_penta_layer_means_five_layers_not_a_pentagonal_lattice(self):
        found, _, _ = penta.detect("a penta-layer graphene stack")
        assert not found
