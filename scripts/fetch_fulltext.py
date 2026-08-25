#!/usr/bin/env python3
"""Retrieve the article text this release does not redistribute.

Publisher full text is not ours to ship, so `data/raw/fulltext_manifest.tsv` records
which documents were used — key, DOI, open-access URL, SHA-256 and character count —
and this script fetches them back and checks each against its hash. A verified fetch
reconstructs byte-identical inputs, which proves provenance more strongly than shipping
the text would.

    python scripts/fetch_fulltext.py            fetch what is missing
    python scripts/fetch_fulltext.py --check    report status, fetch nothing
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "fulltext_manifest.tsv"
DEST = ROOT / "data" / "raw" / "fulltext"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) acv-reproduction/1.0"}


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def rows() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST}")
    with MANIFEST.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def state(row: dict) -> str:
    """present | corrupt | absent, by comparing the local file to its recorded hash."""
    path = DEST / f"{row['paper_key']}.txt"
    if not path.exists():
        return "absent"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "present" if digest == row["sha256"] else "corrupt"


def fetch(row: dict) -> tuple[bool, str]:
    url = row["oa_url"]
    if not url.startswith("http"):
        return False, "no open-access URL recorded"
    try:
        data = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=60).read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)[:60]
    # Manifest strictly enforces checksums against Grobid-converted XML.
    (DEST / f"{row['paper_key']}.source").write_bytes(data)
    return True, f"{len(data) // 1024} KB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report status only")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = rows()
    counts = {"present": 0, "corrupt": 0, "absent": 0}
    absent = []
    for row in manifest:
        s = state(row)
        counts[s] += 1
        if s != "present":
            absent.append(row)

    print(f"  manifest {len(manifest)} documents")
    for k, v in counts.items():
        print(f"    {k:<9} {v}")

    if args.check or not absent:
        if counts["present"] < len(manifest):
            print("\n  text-dependent checks will report NEEDS-FULLTEXT until these resolve")
        sys.exit(0 if counts["present"] == len(manifest) else 1)

    print(f"\n  fetching {len(absent)} …")
    ok = 0
    for row in absent:
        got, note = fetch(row)
        ok += got
        print(f"    {'ok  ' if got else 'FAIL'} {row['paper_key'][:38]:<39} {note}")

    print(f"\n  {ok} of {len(absent)} retrieved as source documents.")
    print("  Convert them to text with the pipeline's own fetch stage, which applies the")
    print("  same parser the manifest hashes were taken from:  python -m acv.cli fetch")


if __name__ == "__main__":
    main()
