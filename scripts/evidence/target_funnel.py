#!/usr/bin/env python3
"""Why do 12 fully-reporting papers yield only 7 recomputable ones?

Replays the exact selection in flows/verification_flow.py:130-200 and prints, per paper,
where each one leaves the funnel. `--inference` additionally resolves a prototype from
stoichiometry via generate.sites_for() instead of the COMPOSITION_PROTOTYPE whitelist.

    PYTHONPATH=src .venv/bin/python evidences/proof/target_funnel.py
    PYTHONPATH=src .venv/bin/python evidences/proof/target_funnel.py --inference
"""
from __future__ import annotations

import sys
from collections import defaultdict

from acv.knowledge import prototypes as P
from acv.pipeline import extract, generate, normalize
from acv.pipeline.report import score_one
from acv.types import ExtractionStatus

MIN_CLAIMS_FOR_CONSENSUS = 3
MAX_DEPARTURE_FROM_CONSENSUS = 0.25
INFER = "--inference" in sys.argv


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def resolvable(reduced: str) -> bool:
    if P.for_composition(reduced) is not None:
        return True
    if not INFER:
        return False
    try:
        return generate.sites_for(reduced) is not None
    except Exception:                                                # noqa: BLE001
        return False


def main() -> None:
    full = [r for r in extract.load()
            if r.status == ExtractionStatus.OK and r.is_pentagonal_2d
            and score_one(r).reproducible_in_principle]
    print(f"prototype resolution: {'INFERENCE (sites_for)' if INFER else 'REGISTRY only'}")
    print(f"papers reporting every required parameter: {len(full)}\n")

    cand, rows = defaultdict(list), []
    for rec in full:
        lat = [c for c in rec.claims
               if normalize.property_kind(c.property).value.startswith("lattice")]
        num = [c for c in lat if isinstance(c.value, (int, float))]
        ok = []
        for c in num:
            red = normalize.reduced_formula(
                normalize.normalize_formula(c.material_formula) or "")
            if red and resolvable(red):
                ok.append(red)
                cand[red].append((rec.paper_key, float(c.value)))
        reason = ("no lattice claim" if not lat else
                  "lattice value is None" if not num else
                  "no prototype" if not ok else "")
        rows.append((rec.paper_key, len(lat), len(num), ok, reason))

    kept = defaultdict(list)
    for red, grp in cand.items():
        vals = sorted(v for _, v in grp)
        cons = vals[len(vals) // 2] if len(vals) >= MIN_CLAIMS_FOR_CONSENSUS else None
        for pk, v in grp:
            if cons and abs(v - cons) / cons > MAX_DEPARTURE_FROM_CONSENSUS:
                continue
            kept[red].append(pk)
    survivors = {p for v in kept.values() for p in v}

    print(f"{'paper':<36}{'latt':>5}{'num':>5}  {'resolved':<20} {'runs?':<6} why not")
    print("-" * 100)
    for pk, nl, nn, ok, reason in rows:
        runs = pk in survivors
        why = reason or ("" if runs else "consensus filter")
        print(f"{pk[:36]:<36}{nl:>5}{nn:>5}  {str(sorted(set(ok)))[:20]:<20} "
              f"{'YES' if runs else 'no':<6} {why}")
    print()
    print(f"RUNS: {len(survivors)} papers, "
          f"{sum(len(v) for v in kept.values())} claims, "
          f"{len(kept)} materials -> {sorted(kept)}")


if __name__ == "__main__":
    main()
