"""Tier 0: Analytical aggregation, reportability scoring, and verification.

Executes code-dependent deterministic scoring of extracted manuscript schemas,
evaluating parameters against strict basis-set convergence requirements 
(plane-wave vs. NAO) to derive final reproducible-in-principle aggregates.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from ..settings import FIGURES, PROCESSED
from ..types import Code, Extraction, ExtractionStatus, ReportabilityScore
from . import validate as validate_mod
from ..settings import _TIER0 as _CFG

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & SETTINGS *********                  
#         Basis-family required fields and code classification bindings.       
# =============================================================================

_rep = (_CFG.get("reportability") or {})

UNIVERSAL_REQUIRED: list[str] = _rep.get("universal_required") or [
    "xc_functional",
    "pseudopotential_type",
    "k_mesh",
    "force_threshold_ev_ang",
    "energy_threshold_ev",
]

PLANE_WAVE_REQUIRED: list[str] = _rep.get("plane_wave_required") or ["plane_wave_cutoff_ev"]
NAO_REQUIRED: list[str] = _rep.get("nao_required") or ["mesh_cutoff_ry", "basis_size"]
TWOD_REQUIRED: list[str] = _rep.get("twod_required") or ["vacuum_spacing_ang"]

def _codes(key: str, fallback: set[Code]) -> set[Code]:
    names = _rep.get(key)
    if not names:
        return fallback
    out = {c for c in Code if c.value in {str(n).lower() for n in names}}
    return out or fallback


PLANE_WAVE_CODES: set[Code] = _codes("plane_wave_codes",
                          {Code.VASP, Code.QUANTUM_ESPRESSO, Code.CASTEP, Code.ABINIT})
NAO_CODES: set[Code] = _codes("nao_codes", {Code.SIESTA, Code.CP2K})


def required_fields(code: Code) -> list[str]:
    fields = UNIVERSAL_REQUIRED + TWOD_REQUIRED
    if code in PLANE_WAVE_CODES:
        fields += PLANE_WAVE_REQUIRED
    elif code in NAO_CODES:
        fields += NAO_REQUIRED
    return fields

# =============================================================================
#                   ********* SCORING & AGGREGATION *********                
#      Deterministic missing-parameter counting and statistical summation.     
# =============================================================================

def score_one(rec: Extraction) -> ReportabilityScore:
    code = rec.method.code.value if rec.method.code.reported else Code.NOT_STATED
    fields = required_fields(code)

    missing = [f for f in fields if not getattr(rec.method, f).reported]
    if not rec.method.code.reported:
        missing.append("code")

    return ReportabilityScore(
        paper_key=rec.paper_key,
        doi=rec.doi,
        code=code.value if hasattr(code, "value") else code,
        fields_required=len(fields) + 1,
        fields_reported=len(fields) + 1 - len(missing),
        missing=missing,
        reproducible_in_principle=not missing,
    )


def _audited(records: list[Extraction]) -> list[Extraction]:
    return [r for r in records
            if r.status == ExtractionStatus.OK and r.is_pentagonal_2d]


def summarise(records: list[Extraction]) -> dict[str, Any]:
    usable = _audited(records)
    scores = [score_one(r) for r in usable]

    missing_counter: Counter[str] = Counter()
    for s in scores:
        missing_counter.update(s.missing)

    elastic_units: Counter[str] = Counter()
    gpa_without_thickness = 0
    for r in usable:
        if r.method.elastic_units_reported.reported:
            elastic_units[str(r.method.elastic_units_reported.value)] += 1
            is_gpa = "GPa" in str(r.method.elastic_units_reported.value)
            if is_gpa and not r.method.thickness_for_gpa_conversion_ang.reported:
                gpa_without_thickness += 1

    ok = [r for r in records if r.status == ExtractionStatus.OK]
    return {
        "n_extraction_records": len(records),
        "n_usable": len(usable),
        "n_extracted_ok": len(ok),
        "n_excluded_not_pentagonal": len(ok) - len(usable),
        "status_breakdown": dict(Counter(r.status.value for r in records)),
        "codes": dict(Counter(
            (r.method.code.value.value if hasattr(r.method.code.value, "value")
             else str(r.method.code.value)) if r.method.code.reported else "NOT STATED"
            for r in usable
        )),
        "reproducible_in_principle": sum(1 for s in scores if s.reproducible_in_principle),
        "mean_fraction_reported": (
            sum(s.fraction_reported for s in scores) / len(scores) if scores else 0.0
        ),
        "most_often_missing": missing_counter.most_common(),
        "elastic_units": dict(elastic_units),
        "gpa_without_stated_thickness": gpa_without_thickness,
    }


def run(records: Optional[list[Extraction]] = None, out_dir: Optional[Path] = None) -> dict[str, Any]:
    from . import extract as extract_mod

    records = records if records is not None else extract_mod.load()
    out_dir = Path(out_dir or PROCESSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = [score_one(r) for r in records if r.status == ExtractionStatus.OK]
    with open(out_dir / "reportability.jsonl", "w", encoding="utf-8") as fh:
        for s in scores:
            fh.write(s.model_dump_json() + "\n")

    stats = summarise(records)

    usable = _audited(records)
    vrep = validate_mod.validate(usable)
    with open(out_dir / "validation_flags.jsonl", "w", encoding="utf-8") as fh:
        for flag in vrep.flags:
            fh.write(json.dumps({
                "paper_key": flag.paper_key, "field": flag.field,
                "value": str(flag.value), "reason": flag.reason,
                "severity": flag.severity,
            }) + "\n")
    stats["validation"] = {
        "n_flagged_records": vrep.n_flagged,
        "n_flags": len(vrep.flags),
        "by_field": vrep.by_field(),
    }
    stats["k_sampling_style"] = dict(Counter(
        validate_mod.k_sampling_kind(r.method.k_mesh.value)
        for r in usable if r.method.k_mesh.reported and r.method.k_mesh.value
    ))

    from ..settings import config_fingerprint

    stats["provenance"] = {
        "config_fingerprint": config_fingerprint(),
        "required_fields": {
            "universal": UNIVERSAL_REQUIRED, "twod": TWOD_REQUIRED,
            "plane_wave": PLANE_WAVE_REQUIRED, "nao": NAO_REQUIRED,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats

# =============================================================================
#                     ********* FIGURE GENERATION *********                    
#          Data visualization of literature reportability omissions.           
# =============================================================================

def plot(stats: dict[str, Any], out_dir: Optional[Path] = None) -> Path:
    """One figure: how often each required parameter goes unreported."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = stats["most_often_missing"]
    if not items:
        raise ValueError("nothing missing to plot")
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    n = stats["n_usable"]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 2))
    ax.barh(labels, [100 * v / n for v in values], color="#c1121f")
    ax.set_xlabel(f"% of papers not reporting  (n = {n})")
    ax.set_title("Reportability gaps — pentagonal 2D materials literature")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "reportability_gaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
