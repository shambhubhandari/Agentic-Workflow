"""Tier 2: Mechanical stability and in-plane elastic tensor derivation.

Computes the 2D stiffness tensor (C11, C12, C22, C66) via finite-difference
strain-stress fitting from DFT-relaxed configurations. All scalar outputs 
are strictly normalized to N/m.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..executors import local

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & SETTINGS *********                  
#          Strain increments and physical unit conversion constants.           
# =============================================================================

STRAINS: tuple[float, ...] = (-0.010, -0.005, -0.0025, 0.0025, 0.005, 0.010)
EV_ANG3_TO_GPA: float = 160.21766208


@dataclass
class StrainPoint:
    component: str
    strain: float
    stress_gpa: Optional[float] = None
    energy_ev: Optional[float] = None
    converged: bool = False
    seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ElasticResult:
    label: str
    cell_height_ang: float
    c11_nm: Optional[float] = None
    c12_nm: Optional[float] = None
    c22_nm: Optional[float] = None
    c66_nm: Optional[float] = None
    poisson: Optional[float] = None
    youngs_nm: Optional[float] = None
    r2: dict[str, float] = field(default_factory=dict)
    points: list[StrainPoint] = field(default_factory=list)
    error: Optional[str] = None

    def to_gpa(self, value_nm: float) -> float:
        """Convert an N/m stiffness back to GPa using the cell height."""
        return value_nm / (self.cell_height_ang * 0.1)

# =============================================================================
#                   ********* KINEMATIC DEFORMATION *********                
#      Symmetric and asymmetric strain application on fixed unit cells.        
# =============================================================================

def _strained(atoms: Any, component: str, epsilon: float) -> Any:
    """Return a copy with one in-plane strain component applied."""
    import numpy as np

    strained = atoms.copy()
    cell = strained.cell.array.copy()
    deformation = np.eye(3)
    if component == "xx":
        deformation[0, 0] += epsilon
    elif component == "yy":
        deformation[1, 1] += epsilon
    elif component == "xy":
        deformation[0, 1] += epsilon / 2
        deformation[1, 0] += epsilon / 2
    else:
        raise ValueError(f"unknown component {component!r}")
    strained.set_cell(cell @ deformation.T, scale_atoms=True)
    return strained

# =============================================================================
#                      ********* ORCHESTRATION *********                     
#         SIE-STA invocation, tensor extraction, and linear regression.        
# =============================================================================

def _read_stress_gpa(out_file: Path) -> Optional[dict[str, float]]:
    """Parse the stress tensor from SIESTA output, in GPa."""
    if not out_file.exists():
        return None
    lines = out_file.read_text(errors="replace").splitlines()
    for line in lines:
        if "Stress tensor Voigt" in line:
            try:
                parts = [float(x) for x in line.split("=")[1].split()]
                return {
                    "xx": parts[0] * 0.1, "yy": parts[1] * 0.1, "zz": parts[2] * 0.1,
                    "yz": parts[3] * 0.1, "xz": parts[4] * 0.1, "xy": parts[5] * 0.1,
                }
            except (ValueError, IndexError):
                continue
    return None


def _fit(strains: list[float], stresses: list[float]) -> tuple[Optional[float], float]:
    """Least-squares slope through the origin-ish, plus R^2."""
    import numpy as np

    if len(strains) < 3:
        return None, 0.0
    x, y = np.array(strains), np.array(stresses)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), (1 - ss_res / ss_tot) if ss_tot else 0.0


def compute(
    atoms: Any,
    label: str,
    ranks: int = 6,
    basis: str = "DZP",
    mesh_cutoff_ry: float = 350.0,
    kgrid: tuple[int, int, int] = (10, 10, 1),
    dry_run: bool = True,
    timeout_s: int = 3600,
) -> ElasticResult:
    """Compute C11, C12, C22 and C66 for one monolayer."""
    height = float(atoms.cell.lengths()[2])
    result = ElasticResult(label=label, cell_height_ang=height)

    stresses: dict[str, dict[str, list[float]]] = {}
    for component in ("xx", "yy", "xy"):
        stresses[component] = {"strain": [], "sxx": [], "syy": [], "sxy": []}
        for epsilon in STRAINS:
            point = StrainPoint(component=component, strain=epsilon)
            strained = _strained(atoms, component, epsilon)
            run_label = f"{label}-{component}{epsilon:+.4f}".replace(".", "p")

            res = local.run(
                strained, label=run_label, dry_run=dry_run, ranks=ranks,
                basis=basis, mesh_cutoff_ry=mesh_cutoff_ry, kgrid=kgrid,
                relax=False, timeout_s=timeout_s,
            )
            point.converged = res.converged
            point.energy_ev = res.energy_ev
            point.seconds = res.seconds
            point.error = res.error

            if not dry_run:
                tensor = _read_stress_gpa(Path(res.workdir) / f"{run_label}.out")
                if tensor:
                    point.stress_gpa = tensor[component]
                    stresses[component]["strain"].append(epsilon)
                    stresses[component]["sxx"].append(tensor["xx"])
                    stresses[component]["syy"].append(tensor["yy"])
                    stresses[component]["sxy"].append(tensor["xy"])
            result.points.append(point)

    if dry_run:
        result.error = "dry run: inputs written, nothing executed"
        return result

    to_nm = height * 0.1
    c11, r11 = _fit(stresses["xx"]["strain"], stresses["xx"]["sxx"])
    c12, r12 = _fit(stresses["xx"]["strain"], stresses["xx"]["syy"])
    c22, r22 = _fit(stresses["yy"]["strain"], stresses["yy"]["syy"])
    c66, r66 = _fit(stresses["xy"]["strain"], stresses["xy"]["sxy"])

    result.c11_nm = c11 * to_nm if c11 is not None else None
    result.c12_nm = c12 * to_nm if c12 is not None else None
    result.c22_nm = c22 * to_nm if c22 is not None else None
    result.c66_nm = c66 * to_nm if c66 is not None else None
    result.r2 = {"c11": r11, "c12": r12, "c22": r22, "c66": r66}

    if result.c11_nm and result.c12_nm is not None:
        result.poisson = result.c12_nm / result.c11_nm
        result.youngs_nm = (
            result.c11_nm - result.c12_nm ** 2 / result.c11_nm
            if result.c11_nm else None
        )
    return result
