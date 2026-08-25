#!/usr/bin/env python3
"""Precision and recall of the extractor against the hand-labelled reference set (M8).

Evaluate pipeline recall metrics against expert reference.
Baseline grounding metrics evaluate independently of extraction bounds.
Excluded records denote N/A applicability or untestable stubs.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/interim/labels_expert_merged.csv"
STUB = re.compile(r"full text missing|only abstract|abstract and (references|bibliography)", re.I)

# Cutoff fields collapse to logical OR across code-specific implementations.
FIELD_MAP = {"cutoff": ("plane_wave_cutoff_ev", "mesh_cutoff_ry")}


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    rows = list(csv.DictReader(open(LABELS, encoding="utf-8")))
    get = lambda r, k: (r.get(k) or "").strip()
    stubs = {r["paper_key"] for r in rows if STUB.search(get(r, "notes"))}

    pipeline: dict[str, dict] = {}
    src = ROOT / "data/interim/extraction/rtx3050_q4_0/union.jsonl"
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            pipeline[rec["paper_key"]] = rec

    per: dict[str, dict[str, int]] = {}
    missing_records = set()
    for r in rows:
        key, field = r["paper_key"], r["field"]
        truth = get(r, "reported").lower()
        if truth == "n/a" or key in stubs:
            continue
        rec = pipeline.get(key)
        if rec is None:
            missing_records.add(key)
            continue
        names = FIELD_MAP.get(field, (field,))
        got = any((rec["method"].get(n) or {}).get("reported") for n in names)
        cell = per.setdefault(field, {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
        if truth == "y":
            cell["tp" if got else "fn"] += 1
        else:
            cell["fp" if got else "tn"] += 1

    print(f"  excluded: {len(stubs)} stub paper(s); "
          f"{sum(1 for r in rows if get(r,'reported').lower()=='n/a')} n/a rows")
    if missing_records:
        print(f"  WARNING: {len(missing_records)} labelled paper(s) absent from "
              f"{src.name}: {sorted(missing_records)[:3]}")
    print(f"\n  {'field':<24}{'TP':>4}{'FN':>4}{'FP':>4}{'TN':>4}"
          f"{'recall':>9}{'95% CI':>16}{'precision':>11}")
    print("  " + "-" * 78)
    tot = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for field, c in per.items():
        for k in tot:
            tot[k] += c[k]
        rec_n = c["tp"] + c["fn"]
        prec_n = c["tp"] + c["fp"]
        rec = c["tp"] / rec_n if rec_n else None
        prec = c["tp"] / prec_n if prec_n else None
        lo, hi = wilson(c["tp"], rec_n)
        print(f"  {field:<24}{c['tp']:>4}{c['fn']:>4}{c['fp']:>4}{c['tn']:>4}"
              f"{(f'{rec:.0%}' if rec is not None else '-'):>9}"
              f"{f'[{lo:.0%}, {hi:.0%}]':>16}"
              f"{(f'{prec:.0%}' if prec is not None else '-'):>11}")
    print("  " + "-" * 78)
    R = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else 0
    P = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else 0
    lo, hi = wilson(tot["tp"], tot["tp"] + tot["fn"])
    print(f"  {'OVERALL':<24}{tot['tp']:>4}{tot['fn']:>4}{tot['fp']:>4}{tot['tn']:>4}"
          f"{R:>8.0%}{f'[{lo:.0%}, {hi:.0%}]':>16}{P:>10.0%}")
    f1 = 2 * P * R / (P + R) if P + R else 0
    print(f"\n  recall {R:.1%}   precision {P:.1%}   F1 {f1:.1%}")
    print(f"  MISSES (fn) = parameters stated in the paper and recorded as absent: {tot['fn']}")


if __name__ == "__main__":
    sys.exit(main())
