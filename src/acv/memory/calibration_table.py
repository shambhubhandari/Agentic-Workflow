"""System: calibration_table module.

Provides strict, deterministic logic and strict typing for calibration_table operations.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..settings import PROCESSED

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & THRESHOLDS *********                
#           Default strict boundaries and file path resolution constants.      
# =============================================================================

TABLE_PATH: Path = PROCESSED / "calibration_table.json"

THRESHOLD_STATISTIC: str = "p90_abs_offset"

DEFAULT_TABLE: dict[str, Optional[float]] = {
    "lattice_a": None,
    "lattice_b": None,
    "lattice_unspecified": None,
    "formation_energy": None,
    "cohesive_energy": None,
    "elastic_c11": None,
    "in_plane_stiffness": None,
}

# =============================================================================
#                     ********* SERIALIZATION *********                    
#          Deterministic I/O operations for JSON-persisted state models.       
# =============================================================================

def load(path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(path or TABLE_PATH)
    if not path.exists():
        return {"offsets": dict(DEFAULT_TABLE), "provenance": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save(table: dict[str, Any], path: Optional[Path] = None) -> Path:
    path = Path(path or TABLE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return path

# =============================================================================
#                   ********* CALIBRATION UPDATES *********                 
#       Statistical threshold extraction and property mutation boundaries.     
# =============================================================================

def update_from_calibration(
    stats: dict[str, Any],
    properties: tuple[str, ...] = ("lattice_a", "lattice_b", "lattice_unspecified"),
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Populate lattice thresholds from a calibration campaign summary."""
    table = load(path)
    lattice = (stats or {}).get("lattice")
    if not lattice:
        log.warning("calibration produced no lattice statistics; table unchanged")
        return table

    threshold = lattice.get(THRESHOLD_STATISTIC)
    if threshold is None:
        log.warning("calibration missing %s; table unchanged", THRESHOLD_STATISTIC)
        return table

    for prop in properties:
        table.setdefault("offsets", {})[prop] = threshold
    table.setdefault("provenance", {})["lattice"] = {
        "statistic": THRESHOLD_STATISTIC,
        "value": threshold,
        "n_values": lattice.get("n_values"),
        "n_structures": stats.get("n_usable"),
        "median_abs_offset": lattice.get("median_abs_offset"),
        "max_abs_offset": lattice.get("max_abs_offset"),
        "settings": stats.get("settings"),
    }
    save(table, path)
    log.info(
        "calibration table: lattice threshold %.4f (%.2f%%) from %d values",
        threshold, 100 * threshold, lattice.get("n_values", 0),
    )
    return table


def offsets(path: Optional[Path] = None) -> dict[str, float]:
    """The mapping consumed by flows.verification_flow.CALIBRATED_OFFSETS."""
    table = load(path)
    return {k: v for k, v in (table.get("offsets") or {}).items() if v is not None}
