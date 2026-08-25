#!/usr/bin/env python3
"""Recompute every value the manuscript reports and report PASS/FAIL for each.

    python scripts/verify.py            human-readable
    python scripts/verify.py --json     machine-readable, written to data/processed/

Exit status is non-zero if any entry fails. Entries needing article text report
NEEDS-FULLTEXT rather than failing: run scripts/fetch_fulltext.py to resolve them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acv import settings as S                      # noqa: E402
from acv.verification import registry              # noqa: E402

# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

MARK = {registry.PASS: "PASS", registry.FAIL: "FAIL",
        registry.NEEDS_TEXT: "NEED", registry.SKIP: "SKIP"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = registry.run()
    tally = Counter(r.status for r in results)

    width = max(len(r.entry.id) for r in results)
    # Group by population: accuracy figures come from the 25 hand-labelled papers, the
    # funnel from the 69-paper corpus. Printing them in one undifferentiated list invites
    # a reader to assume a single denominator.
    order = ("expert-labels", "audited-corpus", "hardware-logs", "tier2-targets", "")
    results = sorted(results, key=lambda r: order.index(r.entry.population)
                     if r.entry.population in order else len(order))
    seen = None
    for r in results:
        if r.entry.population != seen:
            seen = r.entry.population
            print(f"\n  -- {seen or 'other'} --")
        shown = "" if r.observed is None else f"got {r.observed}"
        detail = r.detail or shown
        print(f"  {MARK[r.status]}  {r.entry.id:<{width}}  {str(r.entry.value):>9}"
              f"   {detail}")

    print(f"\n  {len(results)} values: " + "  ".join(
        f"{tally[k]} {k.lower()}" for k in (registry.PASS, registry.FAIL,
                                            registry.NEEDS_TEXT, registry.SKIP) if tally[k]))

    if args.json:
        out = S.PROCESSED / "verification_report.json"
        out.write_text(json.dumps([{
            "id": r.entry.id, "section": r.entry.section, "status": r.status,
            "reported": r.entry.value, "observed": r.observed, "detail": r.detail,
        } for r in results], indent=2), encoding="utf-8")
        print(f"  wrote {out.relative_to(S.PROJECT_ROOT)}")

    if tally[registry.NEEDS_TEXT]:
        print("  run scripts/fetch_fulltext.py to resolve NEED entries")
    sys.exit(1 if tally[registry.FAIL] else 0)


if __name__ == "__main__":
    main()
