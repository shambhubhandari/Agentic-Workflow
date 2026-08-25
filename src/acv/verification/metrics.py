"""System: metrics module.

Provides strict, deterministic logic and strict typing for metrics operations.
"""
from __future__ import annotations

# =============================================================================
#                   ********* VERIFICATION METRICS *********                   
#                        Strict definitions for metrics.                       
# =============================================================================

import csv
import json
import math
from collections import Counter
from pathlib import Path

from .. import settings as S

# Normalize plane-wave and real-space cutoffs to a unified `cutoff` field for scoring.
CUTOFF_FIELDS = {"cutoff": ("plane_wave_cutoff_ev", "mesh_cutoff_ry")}

FIELDS_ALL_CODE_FAMILIES = (
    "code", "xc_functional", "pseudopotential_type", "k_mesh", "cutoff",
    "force_threshold_ev_ang", "energy_threshold_ev", "vacuum_spacing_ang",
)


def load_labels(path: Path | None = None) -> list[dict]:
    path = path or S.LABELS_EXPERT
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_extraction(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["paper_key"]] = record
    return records


def extractor_reported(record: dict, field: str) -> bool:
    names = CUTOFF_FIELDS.get(field, (field,))
    return any((record["method"].get(n) or {}).get("reported") for n in names)


def confusion(labels: list[dict], extraction: dict[str, dict],
              fields: tuple[str, ...] | None = None) -> Counter:
    """Counts over scorable judgements. Rows marked n/a carry no decision and are skipped."""
    counts: Counter = Counter()
    for row in labels:
        truth = (row.get("reported") or "").strip().lower()
        key = row["paper_key"]
        if truth not in ("y", "n") or key not in extraction:
            continue
        if fields is not None and row["field"] not in fields:
            continue
        got = extractor_reported(extraction[key], row["field"])
        if truth == "y":
            counts["tp" if got else "fn"] += 1
        else:
            counts["fp" if got else "tn"] += 1
    return counts


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return 100 * precision, 100 * recall, 100 * f1


def presence(counts: Counter) -> tuple[float, float, float]:
    return _prf(counts["tp"], counts["fp"], counts["fn"])


def absence(counts: Counter) -> tuple[float, float, float]:
    """Same judgements, positive class inverted to 'not reported'."""
    return _prf(counts["tn"], counts["fn"], counts["fp"])


def mcc(counts: Counter) -> float:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / denominator) if denominator else 0.0


def balanced_accuracy(counts: Counter) -> float:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return 50 * (sensitivity + specificity)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on a proportion; the normal approximation is unsafe at this n."""
    if not n:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return 100 * (centre - half), 100 * (centre + half)


def constant_baseline_f1(labels: list[dict]) -> float:
    """Presence-F1 of always answering 'reported' — detects nothing, scores well."""
    scorable = [r for r in labels if (r.get("reported") or "").strip().lower() in ("y", "n")]
    positives = sum(1 for r in scorable if r["reported"].strip().lower() == "y")
    if not scorable:
        return 0.0
    precision = positives / len(scorable)
    return 200 * precision / (precision + 1)


def per_field_recall(labels: list[dict], extraction: dict[str, dict]) -> dict[str, tuple]:
    """(recall %, positives, false positives) per field, ordered by the schema."""
    out = {}
    for field in FIELDS_ALL_CODE_FAMILIES + ("basis_size",):
        counts = confusion(labels, extraction, fields=(field,))
        positives = counts["tp"] + counts["fn"]
        recall = 100 * counts["tp"] / positives if positives else float("nan")
        out[field] = (recall, positives, counts["fp"])
    return out


def configuration_metrics() -> dict[str, dict]:
    """Every statistic in the per-configuration accuracy table, keyed by configuration."""
    labels = load_labels()
    out = {}
    for configuration in S.CONFIGURATIONS:
        path = S.union(configuration)
        if not path.exists():
            continue
        counts = confusion(labels, load_extraction(path))
        pp, pr, pf = presence(counts)
        ap, ar, af = absence(counts)
        out[configuration] = {
            "counts": dict(counts),
            "n": sum(counts.values()),
            "presence": {"precision": pp, "recall": pr, "f1": pf},
            "absence": {"precision": ap, "recall": ar, "f1": af},
            "mcc": mcc(counts),
            "balanced_accuracy": balanced_accuracy(counts),
        }
    return out
