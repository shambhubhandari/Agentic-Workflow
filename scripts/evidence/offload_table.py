# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================
#!/usr/bin/env python3
"""Rebuild the n_ctx x layer-offload table from an ollama server log.

    python3 evidences/proof/offload_table.py evidences/proof/laptop-offload-serve-final.txt
"""
import collections, re, sys

lines = open(sys.argv[1], errors="ignore").read().splitlines()
pairs, ctx = [], None
for line in lines:
    m = re.search(r"n_ctx\s*=\s*(\d+)", line)
    if m:
        ctx = int(m.group(1))
    m2 = re.search(r"offloaded (\d+)/(\d+) layers", line)
    if m2:
        pairs.append((ctx, int(m2.group(1)), int(m2.group(2))))

tab = collections.defaultdict(list)
for c, n, _ in pairs:
    tab[c].append(n)

print(f"{'n_ctx':>8} {'loads':>6}  {'layers on GPU (of 37)':<34} varies?")
print("-" * 62)
for c in sorted(k for k in tab if k):
    v = sorted(set(tab[c]))
    print(f"{c:>8} {len(tab[c]):>6}  {str(v):<34} {'YES' if len(v) > 1 else 'no'}")
allv = sorted({n for _, n, _ in pairs})
print("-" * 62)
print(f"model loads              : {len(pairs)}")
print(f"distinct layer splits    : {len(allv)} -> {allv}")
