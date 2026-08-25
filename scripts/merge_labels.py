#!/usr/bin/env python3
"""Merge the two human labelling batches into one long-format reference set.

Batch 1 (labels_human_wide.csv) carries values; batch 2 (labels_human_batch2.csv) is
y/n/na only. Precision, recall and the reproducibility verdict read `reported` alone,
so the two are equivalent for scoring; only the conversion-loss measurement needs the
values, and that stays scoped to batch 1.
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_sheet import split as _split   # same value/unit parsing as batch 1's expand

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "raw" / "labels_expert"
OUT = DIR / "labels_human.csv"
CANON = {"code": "code", "xc_functional": "xc_functional",
         "pseudopotential": "pseudopotential_type", "k_mesh": "k_mesh",
         "cutoff": "cutoff", "force_threshold": "force_threshold_ev_ang",
         "energy_threshold": "energy_threshold_ev", "vacuum": "vacuum_spacing_ang",
         "basis_size": "basis_size"}
PLANE_WAVE_NA = "--plane-wave-na" in sys.argv   # treat basis_size 'n' as 'na' on PW codes

# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def col(row, name):
    if name in row:
        return row[name]
    for k in row:
        if k and k.lower().split("(")[0].strip().replace(" ", "_") == name:
            return row[k]
    return ""

def rep(cell, field):
    c = (cell or "").strip().lower()
    if not c:
        return ""
    if c in {"n", "no", "-"}:
        return "n"
    if c.replace("/", "").replace(".", "") in {"na", "n a"}:
        return "n/a"
    if field == "basis_size" and c in {"plane", "plane-wave", "planewave", "pw"}:
        return "n/a"
    return "y"

# Codes confirmed by the human labeller. NEVER inferred from the extractor's own output:
# that is the system under test, and it misnames codes -- it reports Quantum ESPRESSO for
# 10.1186_s11671-018-2687-y, which is a DMol3 paper. Inferring from it would let an
# extraction error silently rewrite the reference standard.
HUMAN_CODE = {
    "10.1186_s11671-018-2687-y": "dmol3",   # numerical-orbital basis: basis_size APPLIES
}
PLANE_WAVE = {"vasp", "quantum_espresso", "quantum espresso", "qe", "castep", "abinit"}

pw_codes = set()
if PLANE_WAVE_NA:
    import json
    for line in (ROOT / "data/interim/extraction/rtx3050_q4_0/union_gated.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            key = r["paper_key"]
            code = HUMAN_CODE.get(key)
            if code is None:
                code = str(((r.get("method", {}).get("code") or {}).get("value") or "")).lower()
            if code in PLANE_WAVE:
                pw_codes.add(key)
    pw_codes -= {k for k, v in HUMAN_CODE.items() if v not in PLANE_WAVE}

rows, seen = [], set()
for src in ("labels_human_wide.csv", "labels_human_batch2.csv"):
    path = DIR / src
    if not path.exists():
        continue
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if not any((col(r, f) or "").strip() for f in CANON):
            continue
        if r["paper_key"] in seen:
            continue
        seen.add(r["paper_key"])
        for friendly, canon in CANON.items():
            v = rep(col(r, friendly), canon)
            if PLANE_WAVE_NA and canon == "basis_size" and v == "n" and r["paper_key"] in pw_codes:
                v = "n/a"
            rows.append({"paper_no": r["paper_no"], "paper_key": r["paper_key"],
                         "title": r.get("title", ""), "year": r.get("year", ""),
                         "field": canon, "field_hint": "", "reported": v,
                         # Parse value/unit exactly as batch 1's expand did. Writing the raw
                         # cell here silently breaks check_retrieval, which needs a number.
                         "value": (_split(col(r, friendly), canon)[1]
                                   if src.endswith("wide.csv") else ""),
                         "unit": (_split(col(r, friendly), canon)[2]
                                  if src.endswith("wide.csv") else ""), "evidence_quote": "", "in_retrieved_text": "",
                         "notes": r.get("notes", ""), "batch": "1" if "wide" in src else "2"})

with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"  {len(seen)} papers, {len(rows)} judgements -> {OUT.relative_to(ROOT)}"
      + ("   [plane-wave basis_size -> n/a]" if PLANE_WAVE_NA else ""))
