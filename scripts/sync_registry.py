#!/usr/bin/env python3
"""Check the registry against the manuscript it describes.

The registry is frozen at the values the paper printed; the paper lives elsewhere. Edit
the paper and nothing here notices — `make values` keeps passing against numbers the
manuscript no longer contains, and the audit gate quietly stops measuring anything.

This compares the two and reports the difference. The manuscript path is an argument, so
the tool ships without the paper ever being in this repository.

    python scripts/sync_registry.py --manuscript ../paper
    python scripts/sync_registry.py --manuscript ../paper --update   # rewrite values
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "raw" / "reported_values.yaml"
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def macros(manuscript: Path) -> dict[str, str]:
    """Every generated macro, from numbers.tex wherever it sits under the manuscript."""
    found = {}
    for path in manuscript.rglob("numbers.tex"):
        for name, value in MACRO.findall(path.read_text(encoding="utf-8")):
            found[name] = value.strip()
    return found


def as_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", required=True, type=Path)
    ap.add_argument("--update", action="store_true",
                    help="rewrite registry values to match the manuscript")
    args = ap.parse_args()

    if not args.manuscript.is_dir():
        sys.exit(f"not a directory: {args.manuscript}")
    published = macros(args.manuscript)
    if not published:
        sys.exit(f"no numbers.tex found under {args.manuscript}")

    entries = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or []
    print(f"  manuscript defines {len(published)} macros")
    print(f"  registry holds     {len(entries)} entries\n")

    # A registry entry names a macro only when its id maps onto one; the mapping is by
    # explicit `macro:` key, so prose-only facts are not expected to match.
    changed, checked = [], 0
    for entry in entries:
        name = entry.get("macro")
        if not name:
            continue
        checked += 1
        if name not in published:
            print(f"  GONE      {entry['id']:<40} \\{name} no longer defined")
            continue
        want, have = as_number(published[name]), as_number(entry["value"])
        if want is None or have is None:
            same = str(published[name]) == str(entry["value"])
        else:
            same = abs(want - have) <= float(entry.get("tolerance", 0))
        if not same:
            print(f"  CHANGED   {entry['id']:<40} {entry['value']} -> {published[name]}")
            changed.append((entry, published[name]))

    unmapped = [e["id"] for e in entries if not e.get("macro")]
    print(f"\n  {checked} entries mapped to a macro, {len(unmapped)} prose-only")
    if not changed:
        print("  registry agrees with the manuscript")
        return

    if args.update:
        for entry, value in changed:
            number = as_number(value)
            entry["value"] = number if number is not None else value
        REGISTRY.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
        print(f"  updated {len(changed)} entries — re-run `make values`")
    else:
        print(f"  {len(changed)} entries differ; re-run with --update to adopt them")
        sys.exit(1)


if __name__ == "__main__":
    main()
