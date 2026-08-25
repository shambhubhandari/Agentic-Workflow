"""Tier 0: Structural prototype classification and bounds.

Maintains the static registry of pentagonal MX2 derivatives, their 
expected symmetries, space groups, and strict fractional tolerance 
bounds for symmetry-breaking validation checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
#                  ********* DATA CLASSES & CONSTANTS *********                
#         Schema definitions for symmetry bounds and prototype structures.     
# =============================================================================

@dataclass(frozen=True)
class Prototype:
    name: str
    symmetry: str
    space_group: Optional[str]
    b_over_a: Optional[float]
    tolerance: float
    n_sites: int
    site_pattern: str
    notes: str
    examples: tuple[str, ...] = field(default_factory=tuple)

# =============================================================================
#                     ********* PROTOTYPE REGISTRY *********                   
#        Static mapping of pentagonal architectures and observed compositions. 
# =============================================================================

PROTOTYPES: dict[str, Prototype] = {
    "penta-MX2": Prototype(
        name="penta-MX2",
        symmetry="tetragonal",
        space_group="P-421m",
        b_over_a=1.0,
        tolerance=0.02,
        n_sites=6,
        site_pattern="2 M mid-plane (four-fold), 4 X outer (three-fold)",
        notes="The canonical pentagonal monolayer, derived from one layer of bulk PdSe2.",
        examples=("penta-graphene (C)", "penta-SiC2", "penta-CN2", "penta-SiN2"),
    ),
    "penta-MNX2": Prototype(
        name="penta-MNX2",
        symmetry="orthorhombic",
        space_group=None,
        b_over_a=None,
        tolerance=0.10,
        n_sites=6,
        site_pattern="2 M mid-plane, 4 X outer split 2+2 between two elements",
        notes="Ternary variant. Two different elements on the outer sites break the four-fold axis.",
        examples=("penta-BCP", "penta-BCN", "penta-SiCN"),
    ),
    "penta-PdSe2-type": Prototype(
        name="penta-PdSe2-type",
        symmetry="orthorhombic",
        space_group="Pbca",
        b_over_a=1.025,
        tolerance=0.05,
        n_sites=6,
        site_pattern="2 M mid-plane, 4 X outer, puckered",
        notes="The parent bulk phase, exfoliated. Slightly rectangular geometry (b/a ~ 1.025).",
        examples=("penta-PdSe2", "penta-PdTe2", "penta-PtTe2", "penta-NiP2"),
    ),
}

COMPOSITION_PROTOTYPE: dict[str, str] = {
    "C": "penta-MX2",
    "Si": "penta-MX2",
    "C2Si": "penta-MX2",
    "CN2": "penta-MX2",
    "BN2": "penta-MX2",
    "PdSe2": "penta-PdSe2-type",
    "PdTe2": "penta-PdSe2-type",
    "PtTe2": "penta-PdSe2-type",
    "NiP2": "penta-PdSe2-type",
    "C2B2P2": "penta-MNX2",
}

# =============================================================================
#                    ********* SYMMETRY EVALUATION *********                   
#       Stoichiometric inference logic for expected cell deformations.         
# =============================================================================

def get(name: str) -> Optional[Prototype]:
    return PROTOTYPES.get(name)


def for_composition(reduced_formula: str) -> Optional[Prototype]:
    """Prototype expected for a reduced composition, or None when it is ambiguous."""
    key = COMPOSITION_PROTOTYPE.get(reduced_formula) or _inferred(reduced_formula)
    return PROTOTYPES.get(key) if key else None


def _inferred(reduced_formula: str) -> Optional[str]:
    """Prototype implied by stoichiometry, when the registry has no entry."""
    from ..pipeline.generate import sites_for
    try:
        sites = sites_for(reduced_formula)
    except Exception:
        return None
    if sites is None:
        return None
    return "penta-MNX2" if sites[2] else "penta-MX2"


def expected_symmetry(reduced_formula: str) -> dict[str, Any]:
    """What symmetry a relaxed cell of this composition should retain."""
    proto = for_composition(reduced_formula)
    if proto is None:
        return {
            "prototype": None,
            "symmetry": "unknown",
            "b_over_a": None,
            "tolerance": None,
            "note": (
                f"no prototype recorded for {reduced_formula!r}; a symmetry check would "
                "be an unconstrained assumption"
            ),
        }
    return {
        "prototype": proto.name,
        "symmetry": proto.symmetry,
        "b_over_a": proto.b_over_a,
        "tolerance": proto.tolerance,
        "site_pattern": proto.site_pattern,
        "note": proto.notes,
    }
