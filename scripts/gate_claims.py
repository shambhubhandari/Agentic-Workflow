#!/usr/bin/env python3
"""Apply the claim-value grounding gate to an already-extracted corpus.

Gate application strictly deterministic over values and full text.
the corrected corpus can be derived from an existing `extracted.jsonl` exactly as if the
gate had been in `extract.py` when that run happened. This is what makes the correction
affordable -- re-extracting 69 papers costs hours, this costs seconds and gives the same
answer.

    PYTHONPATH=src .venv/bin/python scripts/gate_claims.py data/interim/extracted.jsonl

Writes <input>.gated.jsonl beside the input and prints a report. The input is never
modified.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acv.pipeline.evaluate import value_grounded_in  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def fulltext(paper_key: str) -> str | None:
    p = ROOT / f"data/raw/fulltext/{paper_key}.txt"
    return p.read_text(errors="replace") if p.exists() else None


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/interim/extraction/rtx3050_q4_0/union.jsonl")
    dst = src.with_suffix(".gated.jsonl")
    if dst.exists():
        sys.exit(f"refusing to overwrite {dst}")

    n = Counter()
    cleared: list[tuple[str, str, float, str]] = []
    out = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = fulltext(rec["paper_key"])
        for claim in (rec.get("claims") or []):
            v = claim.get("value")
            if v is None:
                n["no_value"] += 1
                continue
            if text is None:
                # No text to test against. Not a pass: recorded separately, kept as-is,
                # because clearing on absent evidence would manufacture the absence.
                n["no_fulltext"] += 1
                continue
            if value_grounded_in(float(v), text):
                n["grounded"] += 1
            else:
                n["CLEARED"] += 1
                cleared.append((rec["paper_key"], str(claim.get("property")), float(v),
                                str(claim.get("material_formula"))))
                claim["value"] = None
        out.append(rec)

    dst.write_text("".join(json.dumps(r) + "\n" for r in out))

    total = sum(n.values())
    print(f"  {src}  ->  {dst}")
    print(f"  claims {total}: grounded {n['grounded']}, cleared {n['CLEARED']}, "
          f"no value {n['no_value']}, no full text {n['no_fulltext']}")
    if total - n["no_value"] - n["no_fulltext"]:
        rate = 100 * n["CLEARED"] / (n["grounded"] + n["CLEARED"])
        print(f"  fabrication rate over testable valued claims: {rate:.1f}%")
    print()
    print("  cleared values, and how many distinct papers emitted each:")
    for (prop, val, mat), c in Counter((p, v, m) for _, p, v, m in cleared).most_common():
        papers = len({k for k, p2, v2, _ in cleared if (p2, v2) == (prop, val)})
        print(f"     {prop:<10} {val:<8} {mat:<8} {c:>3} claims across {papers} papers")


if __name__ == "__main__":
    main()
