"""Tier 0.5: Canonicalisation of extracted entity claims.

Maps free-text property names and material strings onto strict controlled 
vocabularies to enable automated cross-referencing against structural databases.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from .penta import ELEMENTS


class PropertyKind(str, Enum):
    LATTICE_A = "lattice_a"
    LATTICE_B = "lattice_b"
    LATTICE_C = "lattice_c"
    LATTICE_UNSPEC = "lattice_unspecified"
    FORMATION_ENERGY = "formation_energy"
    COHESIVE_ENERGY = "cohesive_energy"
    ELASTIC_C11 = "elastic_c11"
    ELASTIC_C12 = "elastic_c12"
    ELASTIC_C22 = "elastic_c22"
    ELASTIC_C66 = "elastic_c66"
    IN_PLANE_STIFFNESS = "in_plane_stiffness"
    POISSON_RATIO = "poisson_ratio"
    BULK_MODULUS = "bulk_modulus"
    YOUNGS_MODULUS = "youngs_modulus"
    PHONON_STABILITY = "phonon_stability"
    OTHER = "other"


# =============================================================================
#                   ********* PROPERTY MAPPERS *********                     
#       Regex dispatch translating free-text claims to PropertyKind enums.   
# =============================================================================

_PROPERTY_PATTERNS: list[tuple[str, PropertyKind]] = [
    (r"lattice.*\ba\b|^a$|lattice_parameter_a|lattice.*constant.*a", PropertyKind.LATTICE_A),
    (r"lattice.*\bb\b|^b$|lattice_parameter_b", PropertyKind.LATTICE_B),
    (r"lattice.*\bc\b|^c$|lattice_parameter_c", PropertyKind.LATTICE_C),
    (r"lattice", PropertyKind.LATTICE_UNSPEC),
    (r"formation.*energ", PropertyKind.FORMATION_ENERGY),
    (r"cohesive.*energ|binding.*energ", PropertyKind.COHESIVE_ENERGY),
    (r"c\s*_?\s*11|c11", PropertyKind.ELASTIC_C11),
    (r"c\s*_?\s*12|c12", PropertyKind.ELASTIC_C12),
    (r"c\s*_?\s*22|c22", PropertyKind.ELASTIC_C22),
    (r"c\s*_?\s*66|c66", PropertyKind.ELASTIC_C66),
    (r"in.?plane.*stiff|2d.*modul|c2d|stiffness", PropertyKind.IN_PLANE_STIFFNESS),
    (r"poisson", PropertyKind.POISSON_RATIO),
    (r"bulk.*modul", PropertyKind.BULK_MODULUS),
    (r"young", PropertyKind.YOUNGS_MODULUS),
    (r"phonon|dynamic.*stab|imaginary", PropertyKind.PHONON_STABILITY),
]


def property_kind(raw: str) -> PropertyKind:
    """Map a free-text property name onto the canonical vocabulary."""
    text = (raw or "").strip().lower().replace("-", " ")
    for pattern, kind in _PROPERTY_PATTERNS:
        if re.search(pattern, text):
            return kind
    return PropertyKind.OTHER


# =============================================================================
#                  ********* STOICHIOMETRY REDUCTION *********               
#      Mathematical reduction of molecular formulas to OPTIMADE standards.   
# =============================================================================

_STRIP_PREFIXES: tuple[str, ...] = ("penta-", "penta ", "p-", "pentagonal ", "2d ", "monolayer ", "bilayer ")


def normalize_formula(raw: Optional[str]) -> Optional[str]:
    """Reduce a claimed material string to a bare chemical formula."""
    if not raw:
        return None

    text = raw.strip()
    changed = True
    while changed:
        changed = False
        lowered = text.lower()
        for prefix in _STRIP_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
                break

    text = re.sub(r"\s*\(.*?\)\s*", " ", text).strip()
    text = re.split(r"\s+(?:monolayer|nanoribbon|nanosheet|sheet|structure)\b", text)[0]
    text = text.strip()

    if not re.fullmatch(r"([A-Z][a-z]?\d*(?:\.\d+)?)+", text):
        return None

    symbols = re.findall(r"([A-Z][a-z]?)", text)
    if not symbols or not all(s in ELEMENTS for s in symbols):
        return None
    return text


def reduced_formula(formula: str) -> Optional[str]:
    """Return the composition-reduced formula matching the OPTIMADE standard."""
    from math import gcd
    from functools import reduce

    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts: dict[str, int] = {}
    for symbol, number in tokens:
        if not symbol:
            continue
        counts[symbol] = counts.get(symbol, 0) + (int(number) if number else 1)
    if not counts:
        return None

    divisor = reduce(gcd, counts.values())
    parts = []
    for symbol in sorted(counts):
        n = counts[symbol] // divisor
        parts.append(symbol if n == 1 else f"{symbol}{n}")
    return "".join(parts)
