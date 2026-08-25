"""Tier 1: MLIP triage of database reference structures.

Applies universal machine learning interatomic potentials (MACE-MP) to 
execute structurally-constrained 2D relaxations on database geometries,
profiling potential accuracy against known DFT endpoints.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from ..settings import PROCESSED
from ..settings import effective as _eff

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & SETTINGS *********                  
#         MACE-MP configuration, relaxation tolerances, and constraints.       
# =============================================================================

MODEL: str = "small"
FMAX: float = 0.02
MAX_STEPS: int = 200
VACUUM_AXIS_MIN_ANG: float = 8.0

@dataclass
class ScreenResult:
    entry_id: str
    provider: str
    formula: str
    nsites: int
    ref_a: float
    ref_b: float
    mlip_a: Optional[float] = None
    mlip_b: Optional[float] = None
    energy_per_atom: Optional[float] = None
    max_force: Optional[float] = None
    rel_error_a: Optional[float] = None
    rel_error_b: Optional[float] = None
    seconds: Optional[float] = None
    status: str = "ok"
    error: Optional[str] = None


def _get_calculator() -> Any:
    from mace.calculators import mace_mp
    return mace_mp(model=MODEL, default_dtype="float64", device="cuda")


# =============================================================================
#                   ********* STRUCTURAL MECHANICS *********                 
#      Frechet cell masking for vacuum-preserving 2D relaxation constraints.   
# =============================================================================

def optimade_to_atoms(entry: dict[str, Any]) -> Any:
    from ase import Atoms
    attrs = entry["attributes"]
    return Atoms(
        symbols=attrs["species_at_sites"],
        positions=attrs["cartesian_site_positions"],
        cell=attrs["lattice_vectors"],
        pbc=True,
    )


def _vacuum_axis(atoms: Any) -> int:
    lengths = atoms.cell.lengths()
    return int(max(range(3), key=lambda i: lengths[i]))


def relax_2d(atoms: Any, calc: Any, fmax: float = FMAX, steps: int = MAX_STEPS) -> Any:
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    atoms = atoms.copy()
    atoms.calc = calc

    vac = _vacuum_axis(atoms)
    mask = [0, 0, 0, 0, 0, 0]
    in_plane = [i for i in range(3) if i != vac]
    for i in in_plane:
        mask[i] = 1
    shear = {frozenset({0, 1}): 5, frozenset({0, 2}): 4, frozenset({1, 2}): 3}
    mask[shear[frozenset(in_plane)]] = 1

    opt = BFGS(FrechetCellFilter(atoms, mask=mask), logfile=None)  # type: ignore[arg-type]
    opt.run(fmax=fmax, steps=steps)
    return atoms


# =============================================================================
#                     ********* MLIP ORCHESTRATION *********                   
#        Stateful relaxation execution and force-field error profiling.        
# =============================================================================

def screen_one(entry: dict[str, Any], provider: str, calc: Any) -> ScreenResult:
    attrs = entry["attributes"]
    atoms = optimade_to_atoms(entry)
    lengths = sorted(atoms.cell.lengths())
    result = ScreenResult(
        entry_id=str(entry.get("id")),
        provider=provider,
        formula=attrs.get("chemical_formula_reduced", ""),
        nsites=len(atoms),
        ref_a=round(float(lengths[0]), 4),
        ref_b=round(float(lengths[1]), 4),
    )
    if lengths[2] < VACUUM_AXIS_MIN_ANG:
        result.status = "failed"
        result.error = "no vacuum axis; not a monolayer"
        return result

    try:
        start = time.time()
        relaxed = relax_2d(atoms, calc)
        new_lengths = sorted(relaxed.cell.lengths())
        forces = relaxed.get_forces()
        result.mlip_a = round(float(new_lengths[0]), 4)
        result.mlip_b = round(float(new_lengths[1]), 4)
        result.energy_per_atom = round(float(relaxed.get_potential_energy()) / len(relaxed), 4)
        result.max_force = round(float((forces**2).sum(axis=1).max() ** 0.5), 4)
        result.rel_error_a = round(float(abs(result.mlip_a - result.ref_a) / result.ref_a), 4)
        result.rel_error_b = round(float(abs(result.mlip_b - result.ref_b) / result.ref_b), 4)
        result.seconds = round(time.time() - start, 1)
    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def run(
    formulas: Optional[list[str]] = None,
    limit_per_formula: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    limit_per_formula = _eff('tier0', 'screen.limit_per_formula', limit_per_formula, 6)
    import requests
    from .crossref import PROVIDERS, MIN_VACUUM_ANG

    out_dir = Path(out_dir or PROCESSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    if formulas is None:
        from . import extract as extract_mod
        from .normalize import normalize_formula, reduced_formula
        from ..types import ExtractionStatus

        seen: set[str] = set()
        for rec in extract_mod.load():
            if rec.status != ExtractionStatus.OK:
                continue
            for claim in rec.claims:
                norm = normalize_formula(claim.material_formula)
                red = reduced_formula(norm) if norm else None
                if red:
                    seen.add(red)
        formulas = sorted(seen)

    log.info("screening compositions: %s", formulas)
    calc = _get_calculator()

    results: list[ScreenResult] = []
    for formula in formulas:
        for provider, base in PROVIDERS.items():
            try:
                response = requests.get(
                    base,
                    params={
                        "filter": f'chemical_formula_reduced="{formula}"',
                        "page_limit": limit_per_formula,
                        "response_fields": "cartesian_site_positions,species_at_sites,"
                                           "lattice_vectors,nsites,chemical_formula_reduced",
                    },
                    timeout=60,
                )
                if response.status_code != 200:
                    continue
                entries = response.json().get("data", [])
            except Exception as exc:
                log.warning("%s query failed for %s: %s", provider, formula, exc)
                continue

            for entry in entries[:limit_per_formula]:
                attrs = entry.get("attributes", {})
                if not attrs.get("cartesian_site_positions"):
                    continue
                lengths = [sum(v * v for v in row) ** 0.5 for row in attrs["lattice_vectors"]]
                if max(lengths) < MIN_VACUUM_ANG:
                    continue
                res = screen_one(entry, provider, calc)
                results.append(res)
                log.info(
                    "  %-28s %-6s n=%-3d  a %.3f->%s  err %s  %ss",
                    res.entry_id[:28], res.formula, res.nsites, res.ref_a,
                    res.mlip_a, res.rel_error_a, res.seconds,
                )

    with open(out_dir / "screen.jsonl", "w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(asdict(res)) + "\n")

    ok = [r for r in results if r.status == "ok" and r.rel_error_a is not None]
    errors = sorted(r.rel_error_a for r in ok if r.rel_error_a is not None)
    stats: dict[str, Any] = {
        "n_screened": len(results),
        "n_ok": len(ok),
        "median_lattice_error_pct": (
            round(100 * float(errors[len(errors) // 2]), 2) if errors else None
        ),
        "worst_lattice_error_pct": round(100 * float(errors[-1]), 2) if errors else None,
        "total_seconds": round(sum(r.seconds or 0 for r in results), 1),
        "model": f"MACE-MP {MODEL}",
    }
    (out_dir / "screen_summary.json").write_text(json.dumps(stats, indent=2))
    log.info("screening: %s", stats)
    return stats
