#!/usr/bin/env python3
"""Wide human labelling sheet <-> the long format the model sheets use.

    human_sheet.py build    15 rows, one per paper, only the cells a person fills
    human_sheet.py expand   -> labels_human.csv, 135 rows, schema-identical to the
                              model sheets so three_way_agreement.py can pair them

Wide is for the human; long is for the analysis. Keeping both means the person types
9 cells per paper instead of filling 12 columns across 9 rows, and nothing downstream
has to know the difference.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "raw" / "labels_expert"
WIDE = DIR / "labels_human_wide.csv"
LONG = DIR / "labels_human.csv"
MODEL = ROOT / "data" / "raw" / "labels_model" / "claude" / "labels_rater2.csv"

# friendly column  ->  canonical field name in the model sheets
FIELDS = {
    "code":              "code",
    "xc_functional":     "xc_functional",
    "pseudopotential":   "pseudopotential_type",
    "k_mesh":            "k_mesh",
    "cutoff":            "cutoff",
    "force_threshold":   "force_threshold_ev_ang",
    "energy_threshold":  "energy_threshold_ev",
    "vacuum":            "vacuum_spacing_ang",
    "basis_size":        "basis_size",
}
NUM = re.compile(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)$")
# Restrict magnitude/unit splitting strictly to physical measurement fields.
NUMERIC = {"cutoff", "force_threshold_ev_ang", "energy_threshold_ev", "vacuum_spacing_ang"}


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def build() -> None:
    rows = list(csv.DictReader(MODEL.open(encoding="utf-8")))
    papers: dict[str, dict] = {}
    for r in rows:
        papers.setdefault(r["paper_key"], r)
    have = {p.stem for p in (DIR / "pdf").glob("*.pdf")}
    # Titles come from the corpus, not the model sheets: those store them truncated,
    # sometimes mid-MathML-tag, which leaves a one-character title after stripping.
    import json
    titles = {}
    corpus = ROOT / "data/raw/corpus/corpus_298_locked.jsonl"
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            titles[re.sub(r"[^a-z0-9._-]", "_", (rec.get("doi") or "").lower())] = rec.get("title", "")

    def clean(t: str) -> str:
        """Titles carry MathML and <sub> tags from the OpenAlex record."""
        # Handle truncation artifacts on closing tags (`>?`).
        t = re.sub(r"<[^>]*>?", "", t or "")
        return re.sub(r"\s+", " ", t).strip()

    with WIDE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["paper_no", "paper_key", "year", "pdf", *FIELDS, "notes", "title"])
        for k, r in sorted(papers.items(), key=lambda x: int(x[1]["paper_no"])):
            w.writerow([r["paper_no"], k, r.get("year", ""),
                        "yes" if k in have else "GET",
                        *[""] * len(FIELDS), "", clean(titles.get(k) or r.get("title"))[:70]])
    print(f"  wrote {WIDE.relative_to(ROOT)}  ({len(papers)} rows, "
          f"{len(FIELDS)} cells each = {len(papers)*len(FIELDS)} judgements)")
    print(f"  {len(have)} of {len(papers)} PDFs present; rows marked GET need fetching")


def split(cell: str, field: str = "") -> tuple[str, str, str]:
    """One typed cell -> (reported, value, unit)."""
    c = cell.strip()
    if not c:
        return "", "", ""
    if c.lower() in {"n", "no", "-"}:
        return "n", "", ""
    if c.lower().replace("/", "").replace(".", "") in {"na", "n a"}:
        return "n/a", "", ""
    if field == "basis_size" and c.lower() in {"plane", "plane-wave", "planewave", "pw"}:
        return "n/a", "", ""   # plane-wave code: no numerical-orbital basis size to state
    if field in NUMERIC:
        sci = _scientific(c)
        if sci is not None:
            return "y", sci[0], sci[1]
        if m := NUM.match(c):
            return "y", m.group(1), m.group(2).strip()
    return "y", c, ""


# Labellers write powers of ten several ways: "1.10^-4" (1 times 10^-4), "1 x 10-5",
# "10-5", "1e-5". Left to the plain numeric regex, "1.10^-4" parses as the number 1.10
# and every downstream check silently looks for the wrong value.
SCI = re.compile(r"""^(?P<mant>[-+]?\d*\.?\d+)?\s*
                      (?:[.x*\u00d7\u22c5]\s*)?
                      10\s*[\^]?\s*(?P<exp>[-\u2212+]?\s*\d+)
                      \s*(?P<unit>.*)$""", re.X)


def _scientific(cell: str) -> tuple[str, str] | None:
    """'1.10^-4' / '1 x 10-5' / '10-5' -> ('1e-04', unit).  None if not this shape."""
    m = SCI.match(cell.replace("\u2212", "-"))
    if not m:
        return None
    mant = float(m.group("mant")) if m.group("mant") else 1.0
    # "1.10^-4" means 1 x 10^-4: the mantissa's ".10" is the "10" of the power, not a
    # decimal fraction. Any mantissa that is exactly n.10 is read that way.
    if m.group("mant") and m.group("mant").endswith(".10"):
        mant = float(m.group("mant")[:-3] or 1)
    exp = int(m.group("exp").replace(" ", ""))
    return f"{mant * 10**exp:g}", m.group("unit").strip()


def _column(row: dict, friendly: str) -> str:
    """Match a column even if the sheet was renamed in a spreadsheet.

    People retitle headers while labelling ("cutoff" -> "cutoff (mesh)"). Matching on
    the leading token keeps that from silently dropping a whole field.
    """
    if friendly in row:
        return row[friendly]
    want = friendly.lower()
    for k in row:
        if k and k.lower().split("(")[0].strip().replace(" ", "_") == want:
            return row[k]
    return ""


def expand() -> None:
    if not WIDE.exists():
        sys.exit(f"missing {WIDE} -- run `human_sheet.py build` first")
    meta = {r["paper_key"]: r for r in csv.DictReader(MODEL.open(encoding="utf-8"))}
    hints = {}
    for r in csv.DictReader(MODEL.open(encoding="utf-8")):
        hints.setdefault(r["field"], r.get("field_hint", ""))

    cols = ["paper_no", "paper_key", "title", "year", "field", "field_hint",
            "reported", "value", "unit", "evidence_quote", "in_retrieved_text", "notes"]
    out, filled = [], 0
    for row in csv.DictReader(WIDE.open(encoding="utf-8")):
        m = meta.get(row["paper_key"], {})
        for friendly, canonical in FIELDS.items():
            rep, val, unit = split(_column(row, friendly), canonical)
            filled += bool(rep)
            out.append({"paper_no": row["paper_no"], "paper_key": row["paper_key"],
                        "title": m.get("title", ""), "year": row.get("year", ""),
                        "field": canonical, "field_hint": hints.get(canonical, ""),
                        "reported": rep, "value": val, "unit": unit,
                        "evidence_quote": "", "in_retrieved_text": "",
                        "notes": row.get("notes", "")})
    with LONG.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"  wrote {LONG.relative_to(ROOT)}  ({len(out)} rows, {filled} filled, "
          f"{len(out)-filled} still blank)")
    if filled < len(out):
        blanks = {r["paper_key"] for r in out if not r["reported"]}
        print(f"  incomplete papers: {len(blanks)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "expand": expand}.get(cmd, lambda: sys.exit(f"unknown: {cmd}"))()
