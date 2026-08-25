"""Tier 0: Detection and classification of pentagonal 2D materials.

Validates extracted plaintext against pentagonal combinatorial naming 
vocabularies, classifying valid structures by their distinct elemental counts.
"""

from __future__ import annotations

import re
from typing import Optional

from ..types import Family

# =============================================================================
#                    ********* CONSTANTS & PATTERNS *********                  
#         Periodic definitions and regex matchers for pentagonal motifs.       
# =============================================================================

ELEMENTS: set[str] = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}

_FORMULA = r"([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){0,3})"
PENTA_FORMULA = re.compile(rf"\b(?i:penta[- ]|p-|pentagonal\s+){_FORMULA}\b")

PENTA_STRUCTURAL = re.compile(
    r"\bpenta[- ]?graphene\b"
    r"|\bcairo[- ](?:tessellat\w+|tiling|tile|lattice|pentagon\w*)\b"
    r"|\bpentagonal\s+(?:monolayer|nanosheet|sheet|bilayer|nanoribbon|nanotube|lattice|"
    r"structure|allotrope|network|framework)\b"
    r"|\bpenta[- ]?monolayer\b"
    r"|\bpentagonal\s+two[- ]dimensional\b",
    re.I,
)

_NOT_FORMULA: set[str] = {
    "graphene", "layer", "layers", "sheet", "sheets", "structure", "structures",
    "monolayer", "monolayers", "material", "materials", "lattice", "ring", "rings",
}

# =============================================================================
#                   ********* STOICHIOMETRIC LOGIC *********                 
#       Validation and elemental separation for generative formula strings.    
# =============================================================================

def split_elements(formula: str) -> list[str]:
    """Return the element symbols in a formula string, in order, or [] if invalid."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    elements = [sym for sym, _ in tokens if sym]
    if not elements or not all(sym in ELEMENTS for sym in elements):
        return []
    return elements

# =============================================================================
#                       ********* CLASSIFICATION *********                   
#      Family categorisation mapping validated formulas to element counts.     
# =============================================================================

def classify(formula: str) -> Family:
    """Classify a penta formula by its count of distinct elements."""
    elements = split_elements(formula)
    if not elements:
        return Family.UNKNOWN
    return {
        1: Family.UNITARY,
        2: Family.BINARY,
        3: Family.TERNARY,
        4: Family.QUATERNARY,
    }.get(len(set(elements)), Family.UNKNOWN)


def detect(text: str) -> tuple[bool, Optional[str], Family]:
    """Detect a pentagonal 2D material in `text`."""
    if not text:
        return False, None, Family.UNKNOWN

    if re.search(r"\bpenta[- ]?graphene\b", text, re.I):
        return True, "C", Family.UNITARY

    for match in PENTA_FORMULA.finditer(text):
        candidate = match.group(1)
        if candidate.lower() in _NOT_FORMULA:
            continue
        elements = split_elements(candidate)
        if elements:
            return True, candidate, classify(candidate)

    if PENTA_STRUCTURAL.search(text):
        return True, None, Family.UNKNOWN

    return False, None, Family.UNKNOWN
