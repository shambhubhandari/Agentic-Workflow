"""Tier 0: Generative structural substitution on pentagonal templates.

Constructs starting geometries for MX2 families via targeted atomic 
substitutions on a 6-site PdSe2 structural archetype, scaling initial 
cells via empirical lattice constants or covalent radius ratios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from ..settings import PROJECT_ROOT

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & SETTINGS *********                  
#         Reference CIF templates, vacuum padding, and chemical spaces.        
# =============================================================================

TEMPLATE_CIF: Path = PROJECT_ROOT / "data" / "raw" / "structures" / "PdSe2.cif"
DEFAULT_VACUUM_ANG: float = 20.0

M_ELEMENTS: list[str] = ["C", "Si", "Ge", "Sn", "B", "Al", "N", "P", "As", "Zn", "Cd", "Ni", "Pd", "Pt"]
X_ELEMENTS: list[str] = ["C", "Si", "Ge", "N", "P", "As", "O", "S", "Se", "Te", "B", "H", "F"]


@dataclass
class Template:
    atoms: Any
    m_indices: list[int]
    x_upper: list[int]
    x_lower: list[int]

    @property
    def n_sites(self) -> int:
        return len(self.m_indices) + len(self.x_upper) + len(self.x_lower)

# =============================================================================
#                   ********* TEMPLATE EXTRACTION *********                  
#      Site identification and atomic stratification of the parent geometry.   
# =============================================================================

def load_template(path: Optional[Path] = None) -> Template:
    """Load the prototype structure and stratify its sites."""
    from ase.io import read

    path = path or TEMPLATE_CIF
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    atoms = read(str(path))
    layer = atoms[atoms.positions[:, 2] < atoms.cell.lengths()[2] / 2].copy()
    layer.center(vacuum=DEFAULT_VACUUM_ANG / 2, axis=2)

    heights = layer.positions[:, 2]
    centre = (heights.max() + heights.min()) / 2
    offsets = heights - centre
    tol = 0.25 * max(abs(offsets).max(), 1e-6)

    m_idx = [i for i, o in enumerate(offsets) if abs(o) <= tol]
    upper = [i for i, o in enumerate(offsets) if o > tol]
    lower = [i for i, o in enumerate(offsets) if o < -tol]

    log.info(
        "template: %d sites (%d M mid-plane, %d/%d X upper/lower)",
        len(layer), len(m_idx), len(upper), len(lower),
    )
    return Template(atoms=layer, m_indices=m_idx, x_upper=upper, x_lower=lower)


def _scale_for_chemistry(atoms: Any, template_symbols: list[str]) -> None:
    """Scale the in-plane cell by the covalent-radius ratio of old to new species."""
    from ase.data import atomic_numbers, covalent_radii

    def mean_radius(symbols: list[str]) -> float:
        return sum(covalent_radii[atomic_numbers[s]] for s in symbols) / len(symbols)

    ratio = mean_radius(atoms.get_chemical_symbols()) / mean_radius(template_symbols)
    cell = atoms.cell.array.copy()
    cell[0] *= ratio
    cell[1] *= ratio
    atoms.set_cell(cell, scale_atoms=True)

# =============================================================================
#                     ********* CELL CONSTRUCTION *********                    
#        Atomic substitution, empirical scaling, and symmetry preservation.    
# =============================================================================

def build(
    m: str,
    x1: str,
    x2: Optional[str] = None,
    template: Optional[Template] = None,
    scale: bool = True,
    target_a: Optional[float] = None,
) -> Any:
    """Build one candidate structure."""
    template = template or load_template()
    atoms = template.atoms.copy()
    original = atoms.get_chemical_symbols()

    symbols = list(original)
    for i in template.m_indices:
        symbols[i] = m

    if x2 is None:
        for i in template.x_upper + template.x_lower:
            symbols[i] = x1
    else:
        for group in (template.x_upper, template.x_lower):
            for n, i in enumerate(sorted(group)):
                symbols[i] = x1 if n % 2 == 0 else x2

    atoms.set_chemical_symbols(symbols)

    if target_a is not None:
        import numpy as np
        current_a = float(np.linalg.norm(atoms.cell.array[0]))
        ratio = target_a / current_a
        cell = atoms.cell.array.copy()
        cell[0] *= ratio
        cell[1] *= ratio
        atoms.set_cell(cell, scale_atoms=True)
    elif scale:
        _scale_for_chemistry(atoms, original)

    atoms.info["penta_design"] = {"m": m, "x1": x1, "x2": x2}
    return atoms

# =============================================================================
#                   ********* STOICHIOMETRIC MAPPING *********                 
#       Algorithmic assignment of elements to structural symmetry sites.       
# =============================================================================

TERNARY_SITES: dict[str, tuple[str, str, str]] = {
    "BCP": ("C", "B", "P"),
}


def sites_for(reduced_formula: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Map a reduced composition onto (M, X1, X2), or None when ambiguous."""
    from ase.formula import Formula

    try:
        counts = Formula(reduced_formula).count()
    except Exception:
        return None
    if not counts:
        return None

    if len(counts) == 1:
        only = next(iter(counts))
        return only, only, None

    if len(counts) == 2:
        (a, na), (b, nb) = sorted(counts.items(), key=lambda kv: kv[1])
        if nb != 2 * na:
            return None
        return a, b, None

    if len(counts) == 3:
        return TERNARY_SITES.get("".join(sorted(counts)))

    return None


def from_formula(reduced_formula: str, target_a: Optional[float] = None) -> Any:
    """Build the penta structure for a reduced composition."""
    sites = sites_for(reduced_formula)
    if sites is None:
        raise ValueError(
            f"cannot place {reduced_formula!r} on the penta prototype: stoichiometry is "
            "not M:X = 1:2 and no site assignment is recorded in TERNARY_SITES"
        )
    m, x1, x2 = sites
    return build(m, x1, x2, target_a=target_a)


def name(m: str, x1: str, x2: Optional[str] = None) -> str:
    return f"penta-{m}{x1}{x2}" if x2 else f"penta-{m}{x1}2"


def enumerate_candidates(
    m_elements: Optional[list[str]] = None,
    x_elements: Optional[list[str]] = None,
    ternary: bool = False,
) -> Iterator[tuple[str, str, Optional[str]]]:
    """Yield (m, x1, x2) combinations across the design space."""
    m_elements = m_elements or M_ELEMENTS
    x_elements = x_elements or X_ELEMENTS

    for m in m_elements:
        for x1 in x_elements:
            if not ternary:
                yield (m, x1, None)
            else:
                for x2 in x_elements:
                    if x2 <= x1:
                        continue
                    yield (m, x1, x2)


def write(atoms: Any, path: Path) -> Path:
    """Write a candidate as a VASP POSCAR."""
    from ase.io import write as ase_write

    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms, format="vasp", direct=True, sort=True)
    return path
