"""Tier 0: Historical literature parameter aggregation.

Aggregates extracted computational methods and parameters from the 
corpus, providing statistical priors (distributions and median bounds) 
for parameter reproduction without predictive inference.
"""

from __future__ import annotations

import logging
import statistics
from functools import lru_cache
from typing import Any, Optional

log = logging.getLogger(__name__)

# =============================================================================
#                  ********* AGGREGATION PARAMETERS *********                 
#          Selected computational fields for literature summarisation.         
# =============================================================================

METHOD_FIELDS: tuple[str, ...] = (
    "code", "xc_functional", "plane_wave_cutoff_ev", "mesh_cutoff_ry",
    "k_mesh", "vacuum_spacing_ang", "force_threshold_ev_ang", "basis_size",
)

# =============================================================================
#                     ********* CORPUS FILTERING *********                    
#          Strict loading and stoichiometry normalization pipelines.           
# =============================================================================

@lru_cache(maxsize=1)
def _records() -> list[Any]:
    from ..pipeline import extract
    from ..types import ExtractionStatus

    return [
        r for r in extract.load()
        if r.status == ExtractionStatus.OK and r.is_pentagonal_2d
    ]


def _reduced(raw: Optional[str]) -> Optional[str]:
    from ..pipeline.normalize import normalize_formula, reduced_formula

    norm = normalize_formula(raw)
    return reduced_formula(norm) if norm else None

# =============================================================================
#                   ********* STATISTICAL SUMMATION *********                 
#       Distribution extraction and bounding for field-wide parameters.        
# =============================================================================

def methods_for(reduced_formula: str) -> dict[str, Any]:
    """Return per-field setting distributions for a target composition."""
    matching = []
    for rec in _records():
        formulas = {_reduced(c.material_formula) for c in rec.claims}
        if reduced_formula in formulas:
            matching.append(rec)

    if not matching:
        return {"formula": reduced_formula, "n_papers": 0,
                "note": "no papers in the corpus report this composition"}

    summary: dict[str, Any] = {"formula": reduced_formula, "n_papers": len(matching)}
    for field in METHOD_FIELDS:
        values = []
        for rec in matching:
            entry = getattr(rec.method, field, None)
            if entry is not None and entry.reported and entry.value is not None:
                v = entry.value
                values.append(getattr(v, "value", v))
        if not values:
            summary[field] = {"reported_by": 0, "of": len(matching)}
            continue

        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        entry: dict[str, Any] = {"reported_by": len(values), "of": len(matching)}
        if numeric:
            entry.update(
                median=round(statistics.median(numeric), 3),
                min=round(min(numeric), 3),
                max=round(max(numeric), 3),
            )
        else:
            from collections import Counter
            entry["values"] = dict(Counter(str(v) for v in values).most_common(4))
        summary[field] = entry
    return summary


def claims_for(reduced_formula: str, property_kind: Optional[str] = None) -> dict[str, Any]:
    """Return every published value and evidence excerpt for a composition."""
    from ..pipeline.normalize import property_kind as classify

    out = []
    for rec in _records():
        for c in rec.claims:
            if _reduced(c.material_formula) != reduced_formula:
                continue
            kind = classify(c.property).value
            if property_kind and not kind.startswith(property_kind):
                continue
            out.append({
                "paper": rec.paper_key,
                "doi": rec.doi,
                "property": c.property,
                "kind": kind,
                "value": c.value,
                "unit": c.unit,
                "material_as_written": c.material_formula,
                "evidence": (c.evidence or "")[:160],
            })
    values = [r["value"] for r in out if isinstance(r["value"], (int, float))]
    return {
        "formula": reduced_formula,
        "n_claims": len(out),
        "spread": (
            {"min": min(values), "max": max(values),
             "median": round(statistics.median(values), 4),
             "range_pct": round(100 * (max(values) - min(values)) / min(values), 1)}
            if len(values) > 1 and min(values) else None
        ),
        "claims": out,
    }


def corpus_summary() -> dict[str, Any]:
    """Coverage of the mined corpus, for orientation."""
    from collections import Counter

    recs = _records()
    comps = Counter(
        f for r in recs for f in {_reduced(c.material_formula) for c in r.claims} if f
    )
    return {
        "n_papers": len(recs),
        "n_claims": sum(len(r.claims) for r in recs),
        "compositions": dict(comps.most_common()),
        "codes": dict(Counter(
            (r.method.code.value.value if hasattr(r.method.code.value, "value")
             else str(r.method.code.value)) if r.method.code.reported else "not_stated"
            for r in recs
        )),
    }
