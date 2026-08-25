"""Tier 2: Basis-set offset calibration for pentagonal geometries.

Relaxes plane-wave reference structures using SIESTA under standardized 
settings to empirically measure the baseline structural offset. This 
offset distribution dictates the verdict threshold for Tier 2 evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from ..settings import INTERIM, PROCESSED
from ..executors import local
from ..settings import effective as _eff

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & SETTINGS *********                  
#        Standardized convergence criteria for baseline structure relaxation.  
# =============================================================================

CHECKPOINT_DIR = INTERIM / "calibration"
MIN_VACUUM_ANG = 8.0
MIN_VACUUM_GAP_ANG = 5.0
PENTA_N_SITES = 6
MIN_PUCKER_ANG = 0.5
MAX_LAYER_SPAN_ANG = 3.0

MIN_DISTINCT_STRUCTURES = int(_eff('tier2', 'calibration.min_distinct_structures', None, 6))

STANDARD = dict(basis="DZP", mesh_cutoff_ry=350.0, kgrid=(10, 10, 1))

@dataclass
class CalibrationPoint:
    entry_id: str
    provider: str
    formula: str
    n_atoms: int
    ref_a: float
    ref_b: float
    siesta_a: Optional[float] = None
    siesta_b: Optional[float] = None
    energy_ev: Optional[float] = None
    offset_a: Optional[float] = None
    offset_b: Optional[float] = None
    converged: bool = False
    seconds: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
#                  ********* STRUCTURE PROCUREMENT *********                 
#      Queries databases for prototypical pentagonal 2D reference materials. 
# =============================================================================

def is_monolayer(attrs: dict[str, Any]) -> bool:
    """Verifies geometric isolation of the layer along its longest axis."""
    vectors = attrs.get("lattice_vectors")
    positions = attrs.get("cartesian_site_positions")
    if not vectors or not positions:
        return False

    lengths = [sum(v * v for v in row) ** 0.5 for row in vectors]
    axis = max(range(3), key=lambda i: lengths[i])
    if lengths[axis] < MIN_VACUUM_ANG:
        return False

    coords = sorted(p[axis] for p in positions)
    gaps = [b - a for a, b in zip(coords, coords[1:])]
    gaps.append(lengths[axis] - coords[-1] + coords[0])
    return max(gaps) >= MIN_VACUUM_GAP_ANG


def is_penta_prototype(attrs: dict[str, Any]) -> bool:
    """Validates structure against the 6-site PdSe2-derived pentagonal motif."""
    from ..knowledge.prototypes import COMPOSITION_PROTOTYPE
    from .normalize import normalize_formula, reduced_formula

    reduced = reduced_formula(
        normalize_formula(attrs.get("chemical_formula_reduced") or "") or ""
    )
    if reduced not in COMPOSITION_PROTOTYPE:
        return False

    positions = attrs.get("cartesian_site_positions") or []
    vectors = attrs.get("lattice_vectors")
    if len(positions) != PENTA_N_SITES or not vectors:
        return False

    lengths = [sum(v * v for v in row) ** 0.5 for row in vectors]
    axis = max(range(3), key=lambda i: lengths[i])
    heights = sorted(p[axis] for p in positions)
    span = heights[-1] - heights[0]
    if not (MIN_PUCKER_ANG <= span <= MAX_LAYER_SPAN_ANG):
        return False

    mid = (heights[0] + heights[-1]) / 2.0
    band = span * 0.2
    return (
        len([z for z in heights if abs(z - mid) < band]) == 2
        and len([z for z in heights if z - mid >= band]) == 2
        and len([z for z in heights if mid - z >= band]) == 2
    )


def candidates(formulas: list[str], max_atoms: int = 8, per_formula: int = 3) -> list[tuple[dict[str, Any], str]]:
    """Fetches candidates via OPTIMADE and filters for pentagonal structural geometry."""
    from .crossref import PROVIDERS

    found: list[tuple[dict[str, Any], str]] = []
    for formula in formulas:
        for provider, base in PROVIDERS.items():
            try:
                response = requests.get(
                    base,
                    params={
                        "filter": f'chemical_formula_reduced="{formula}"',
                        "page_limit": 12,
                        "response_fields": "cartesian_site_positions,species_at_sites,"
                                           "lattice_vectors,nsites,chemical_formula_reduced",
                    },
                    timeout=60,
                )
                if response.status_code != 200:
                    continue
                kept = 0
                for entry in response.json().get("data", []):
                    attrs = entry.get("attributes", {})
                    if not attrs.get("cartesian_site_positions"):
                        continue
                    if not (2 <= (attrs.get("nsites") or 0) <= max_atoms):
                        continue
                    if not is_monolayer(attrs):
                        continue
                    if not is_penta_prototype(attrs):
                        continue
                    found.append((entry, provider))
                    kept += 1
                    if kept >= per_formula:
                        break
            except requests.RequestException as exc:
                log.warning("%s query failed for %s: %s", provider, formula, exc)
    return found


# =============================================================================
#                  ********* CAMPAIGN EXECUTION *********                    
#       Stateful execution of SIESTA relaxations across the reference set.   
# =============================================================================

def _checkpoint(point: CalibrationPoint) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = point.entry_id.replace("/", "_")
    (CHECKPOINT_DIR / f"{safe}.json").write_text(
        json.dumps(asdict(point), indent=2), encoding="utf-8"
    )


def _load_checkpoints() -> dict[str, CalibrationPoint]:
    if not CHECKPOINT_DIR.exists():
        return {}
    out = {}
    for path in CHECKPOINT_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        out[data["entry_id"]] = CalibrationPoint(**data)
    return out


def run_point(entry: dict[str, Any], provider: str, ranks: int = 6, timeout_s: int = 3600) -> CalibrationPoint:
    """Relaxes a single reference structure in SIESTA and measures baseline offset."""
    from .screen import optimade_to_atoms

    attrs = entry["attributes"]
    atoms = optimade_to_atoms(entry)
    lengths = sorted(float(x) for x in atoms.cell.lengths())
    point = CalibrationPoint(
        entry_id=str(entry.get("id")),
        provider=provider,
        formula=attrs.get("chemical_formula_reduced", ""),
        n_atoms=len(atoms),
        ref_a=round(lengths[0], 4),
        ref_b=round(lengths[1], 4),
    )

    label = f"calib-{point.entry_id.replace('/', '_')[:34]}"
    try:
        result = local.run(
            atoms, label=label, dry_run=False, ranks=ranks,
            timeout_s=timeout_s, **STANDARD,
        )
        point.converged = bool(result.converged)
        point.energy_ev = result.energy_ev
        point.seconds = result.seconds
        point.error = result.error

        out_file = Path(result.workdir) / f"{label}.out"
        if out_file.exists():
            for line in out_file.read_text(errors="replace").splitlines():
                if "outcell: Cell vector modules" in line:
                    try:
                        cell = sorted(float(x) for x in line.rsplit(":", 1)[1].split())
                        point.siesta_a, point.siesta_b = round(cell[0], 4), round(cell[1], 4)
                    except (ValueError, IndexError):
                        pass
        if point.siesta_a and point.ref_a:
            point.offset_a = round((point.siesta_a - point.ref_a) / point.ref_a, 5)
        if point.siesta_b and point.ref_b:
            point.offset_b = round((point.siesta_b - point.ref_b) / point.ref_b, 5)
    except Exception as exc:                       # noqa: BLE001
        point.error = f"{type(exc).__name__}: {exc}"

    _checkpoint(point)
    return point


def summarise(points: list[CalibrationPoint]) -> dict[str, Any]:
    """Calculates distribution statistics defining the verdict tolerance."""
    import statistics

    usable = [
        p for p in points
        if p.converged and p.offset_a is not None and p.offset_b is not None
    ]
    offsets = [abs(p.offset_a) for p in usable] + [abs(p.offset_b) for p in usable]
    if not offsets:
        return {"n_points": len(points), "n_usable": 0, "lattice": None}

    distinct = sorted({p.formula for p in usable})
    if len(distinct) < MIN_DISTINCT_STRUCTURES:
        return {
            "n_points": len(points),
            "n_usable": len(usable),
            "distinct_structures": distinct,
            "settings": {**STANDARD, "kgrid": list(STANDARD["kgrid"])},
            "lattice": None,
            "usable": False,
            "note": (
                f"only {len(distinct)} distinct pentagonal structure(s) calibrated; "
                f"{MIN_DISTINCT_STRUCTURES} required before a threshold is defined. "
                "Tier 2 verdicts are INCONCLUSIVE until the reference set is enlarged."
            ),
        }

    offsets.sort()
    return {
        "n_points": len(points),
        "n_usable": len(usable),
        "distinct_structures": distinct,
        "usable": True,
        "settings": {**STANDARD, "kgrid": list(STANDARD["kgrid"])},
        "lattice": {
            "median_abs_offset": round(statistics.median(offsets), 5),
            "mean_abs_offset": round(statistics.fmean(offsets), 5),
            "p90_abs_offset": round(offsets[int(0.9 * (len(offsets) - 1))], 5),
            "max_abs_offset": round(offsets[-1], 5),
            "n_values": len(offsets),
        },
    }


def run(
    formulas: Optional[list[str]] = None,
    ranks: int = 6,
    max_structures: Optional[int] = None,
    deadline_epoch: Optional[float] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Executes the stateful calibration campaign."""
    max_structures = _eff('tier2', 'calibration.max_structures', max_structures, 20)
    out_dir = Path(out_dir or PROCESSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    if formulas is None:
        from ..knowledge.prototypes import COMPOSITION_PROTOTYPE
        formulas = sorted(COMPOSITION_PROTOTYPE)
        
    done = _load_checkpoints()
    log.info("calibration: %d structures already checkpointed", len(done))

    entries = candidates(formulas)
    log.info("calibration: %d candidate structures fetched", len(entries))

    points = list(done.values())
    for entry, provider in entries:
        if len(points) >= max_structures:
            break
        entry_id = str(entry.get("id"))
        if entry_id in done:
            continue
        if deadline_epoch and time.time() > deadline_epoch:
            log.warning("calibration: deadline reached, stopping with %d points", len(points))
            break

        point = run_point(entry, provider, ranks=ranks)
        points.append(point)
        log.info(
            "  %-34s %-6s n=%-2d ref_a=%.3f siesta_a=%s offset=%s  %ss",
            point.entry_id[:34], point.formula, point.n_atoms, point.ref_a,
            point.siesta_a, point.offset_a, point.seconds,
        )

    stats = summarise(points)
    (out_dir / "calibration.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with open(out_dir / "calibration_points.jsonl", "w", encoding="utf-8") as fh:
        for p in points:
            fh.write(json.dumps(asdict(p)) + "\n")
    log.info("calibration: %s", stats)
    return stats
