#!/usr/bin/env python3
"""Field-decision agreement between two extraction passes.

    python3 evidences/proof/pass_agreement.py A.jsonl B.jsonl

A "field decision" is the boolean `method.<field>.reported`. Agreement is the fraction
of those booleans identical across the two passes, over the papers both contain.
"""
import json, sys

# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def load(path):
    out = {}
    for line in open(path):
        if line.strip():
            rec = json.loads(line)
            out[rec["paper_key"]] = rec
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(set(a) & set(b))
agree = dis = 0
differing = []
for k in shared:
    ma = a[k].get("method") or {}
    mb = b[k].get("method") or {}
    n = 0
    for f in set(ma) | set(mb):
        ra = bool((ma.get(f) or {}).get("reported"))
        rb = bool((mb.get(f) or {}).get("reported"))
        if ra == rb:
            agree += 1
        else:
            dis += 1
            n += 1
    if n:
        differing.append((k, n,
                          sum(1 for f in ma if (ma.get(f) or {}).get("reported")),
                          sum(1 for f in mb if (mb.get(f) or {}).get("reported"))))
tot = agree + dis
print(f"{sys.argv[1].split('/')[-1]}  vs  {sys.argv[2].split('/')[-1]}")
print(f"  papers compared      : {len(shared)}")
print(f"  field decisions      : {agree}/{tot} agree = {100*agree/tot:.1f}%")
print(f"  papers with any diff : {len(differing)}/{len(shared)}")
for k, n, na, nb in differing:
    print(f"     {k:<40} {n:>2} field(s)   A={na:>2}  B={nb:>2}")
