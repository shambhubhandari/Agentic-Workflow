"""Tier 0: Physical plausibility gating on extracted parameters.

Evaluates raw extracted schemas against domain-specific physical boundaries,
flagging combinatorial mismatches (e.g., NAO cutoffs for plane-wave codes) 
and bounding impossible scalars before Tier 1 testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..types import Code, Extraction
from ..settings import _TIER0 as _CFG

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & BOUNDARIES *********                  
#        Physics plausibility limits and functional vocabulary definitions.    
# =============================================================================

_cfg_ranges = ((_CFG.get("validation") or {}).get("ranges") or {})

RANGES: dict[str, tuple[float, float]] = {
    "plane_wave_cutoff_ev": (20.0, 2000.0),
    "augmentation_cutoff_ev": (100.0, 5000.0),
    "mesh_cutoff_ry": (50.0, 2000.0),
    "pao_energy_shift_ev": (0.0001, 1.0),
    "basis_split_norm": (0.0, 1.0),
    "vacuum_spacing_ang": (5.0, 60.0),
    "force_threshold_ev_ang": (1e-4, 1.0),
    "energy_threshold_ev": (1e-12, 1e-1),
    "thickness_for_gpa_conversion_ang": (1.0, 30.0),
}
RANGES.update({k: (float(v[0]), float(v[1])) for k, v in _cfg_ranges.items()
               if isinstance(v, (list, tuple)) and len(v) == 2})

KNOWN_FUNCTIONALS: set[str] = {
    "pbe", "pbesol", "lda", "pw91", "blyp", "b3lyp", "hse", "hse06", "hse03",
    "scan", "r2scan", "rscan", "tpss", "revpbe", "rpbe", "optb86b", "optb88",
    "vdw-df", "vdw-df2", "pbe0", "gga", "gga-pbe", "am05", "wc",
}

PLANE_WAVE_CODES: set[Code] = {Code.VASP, Code.QUANTUM_ESPRESSO, Code.CASTEP, Code.ABINIT}
NAO_CODES: set[Code] = {Code.SIESTA, Code.CP2K}


@dataclass
class Flag:
    paper_key: str
    field: str
    value: object
    reason: str
    severity: str = "warn"


@dataclass
class ValidationReport:
    n_records: int = 0
    n_flagged: int = 0
    flags: list[Flag] = field(default_factory=list)

    def by_field(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.flags:
            counts[f.field] = counts.get(f.field, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

# =============================================================================
#                     ********* HEURISTIC PARSERS *********                  
#         Text normalization and classification for k-space sampling strings.  
# =============================================================================

def k_sampling_kind(text: str) -> str:
    """Classify a k-mesh string as 'grid', 'spacing' or 'unparseable'."""
    import re
    raw = str(text)
    low = raw.lower()
    if any(mark in low for mark in ("spacing", "density")) or re.search(
        r"\u00c5\s*[\u207b-]\s*1|a\s*\^?-1|\u00c5\u207b\u00b9", low
    ):
        return "spacing"
    if re.search(r"\d+\s*[x\u00d7*\s]\s*\d+\s*[x\u00d7*\s]\s*\d+", raw):
        return "grid"
    return "unparseable"


def _parse_k_mesh(text: str) -> Optional[tuple[int, int, int]]:
    """Parse mesh strings into a strictly bounded integer triple."""
    import re
    if k_sampling_kind(text) != "grid":
        return None
    match = re.search(
        r"(\d+)\s*[x\u00d7*\s]\s*(\d+)\s*[x\u00d7*\s]\s*(\d+)", str(text)
    )
    if not match:
        return None
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]

# =============================================================================
#                       ********* VALIDATION LOGIC *********                   
#      Combinatorial consistency checks across structured paper extractions.     
# =============================================================================

def validate_one(rec: Extraction) -> list[Flag]:
    """Return plausibility flags for a single extraction."""
    flags: list[Flag] = []
    m = rec.method
    key = rec.paper_key

    for name, (lo, hi) in RANGES.items():
        f = getattr(m, name, None)
        if f is None or not f.reported or f.value is None:
            continue
        try:
            v = float(f.value)
        except (TypeError, ValueError):
            flags.append(Flag(key, name, f.value, "non-numeric value", "error"))
            continue
        if not (lo <= v <= hi):
            flags.append(
                Flag(key, name, v, f"outside plausible range [{lo}, {hi}]", "error")
            )

    if m.k_mesh.reported and m.k_mesh.value:
        kind = k_sampling_kind(m.k_mesh.value)
        if kind == "spacing":
            pass
        elif kind == "unparseable":
            flags.append(Flag(key, "k_mesh", m.k_mesh.value, "unparseable sampling"))
        else:
            mesh = _parse_k_mesh(m.k_mesh.value)
            if mesh is None:
                flags.append(Flag(key, "k_mesh", m.k_mesh.value, "unparseable grid"))
            elif any(n < 1 for n in mesh):
                flags.append(Flag(key, "k_mesh", mesh, "zero or negative divisions", "error"))
            elif min(mesh) > 3:
                flags.append(
                    Flag(key, "k_mesh", mesh,
                         "no sparse axis; does not look like a 2D calculation")
                )

    if m.xc_functional.reported and m.xc_functional.value:
        name = str(m.xc_functional.value).strip().lower().replace(" ", "")
        if not any(k in name for k in KNOWN_FUNCTIONALS):
            flags.append(
                Flag(key, "xc_functional", m.xc_functional.value, "unrecognised functional")
            )

    if m.code.reported:
        code = m.code.value
        if code in PLANE_WAVE_CODES and m.mesh_cutoff_ry.reported:
            flags.append(
                Flag(key, "mesh_cutoff_ry", m.mesh_cutoff_ry.value,
                     f"NAO parameter reported for plane-wave code {code}", "error")
            )
        if code in NAO_CODES and m.plane_wave_cutoff_ev.reported:
            flags.append(
                Flag(key, "plane_wave_cutoff_ev", m.plane_wave_cutoff_ev.value,
                     f"plane-wave cutoff reported for NAO code {code}", "error")
            )

    if m.elastic_units_reported.reported:
        units = str(m.elastic_units_reported.value)
        if "GPA" in units.upper() and not m.thickness_for_gpa_conversion_ang.reported:
            flags.append(
                Flag(key, "thickness_for_gpa_conversion_ang", None,
                     "elastic constants in GPa with no stated monolayer thickness")
            )

    return flags


def validate(records: list[Extraction]) -> ValidationReport:
    """Validate every usable extraction."""
    report = ValidationReport(n_records=len(records))
    for rec in records:
        flags = validate_one(rec)
        if flags:
            report.n_flagged += 1
            report.flags.extend(flags)
    log.info(
        "validation: %d/%d records flagged, %d flags total",
        report.n_flagged, report.n_records, len(report.flags),
    )
    return report
