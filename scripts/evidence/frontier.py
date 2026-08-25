#!/usr/bin/env python3
"""Cost-accuracy frontier over the four measured configurations.

Every constant below is MEASURED, not estimated, and each carries the artefact it came
from. Nothing here is a model of the hardware -- it is the hardware's own accounting.

    python3 evidences/proof/frontier.py                # uses the recorded timings
    python3 evidences/proof/frontier.py logs/          # re-derives timings from logs/

VRAM is sized to the corpus's WIDEST window (28,672 tokens, 3 of 69 papers), not the
median. Sizing to the median under-provisions those three and forces the partial-
residency regime that Evidence 1 shows is the cause of run-to-run instability.
"""
from __future__ import annotations

import os
import re
import sys
import glob

MIB_PER_GIB = 1024.0

# --- measured: `load_tensors: CUDA0 model buffer size` in the ollama journal ---------
WEIGHTS_MIB = {"Q4_K_M": 2375.91, "Q8_0": 4076.43}

# --- measured: `llama_kv_cache: size = ... ( 28672 cells, ...)` in the ollama journal
# The widest window the corpus actually requests. Full table in
# evidences/proof/t4-f16-vs-q4-detail.txt section 4.
KV_MIB_AT_28672 = {"q4_0": 1134.0, "f16": 4032.0}

# --- measured: `sched_reserve: CUDA0 compute buffer size` -----------------------------
COMPUTE_MIB = 182.10

# --- measured: scripts/score_handlabels.py against data/handlabels/labels.csv ---------
# Accuracy is deliberately NOT tabulated here. These constants were scored against a
# model-generated reference set that the study no longer uses; reproducing them beside
# the current figures would put superseded numbers back into circulation. Accuracy comes
# from the expert labels via `make values`, which recomputes it rather than quoting it.
# What this report uniquely contributes is the memory and timing frontier below.

# --- measured: mean of the per-paper seconds in the first pass of each run ------------
# Laptop figure is the median of three full passes (logs/tier0-*.log on that machine).
SECONDS_PER_PAPER = {
    ("Q4_K_M", "q4_0", "laptop"): 120.0,
    ("Q4_K_M", "q4_0", "T4"): 58.3,
    ("Q4_K_M", "f16", "T4"): 42.7,
    ("Q8_0", "f16", "T4"): 49.9,
}

RESIDENCY = {
    ("Q4_K_M", "q4_0", "laptop"): "PARTIAL (23-37 of 37)",
    ("Q4_K_M", "q4_0", "T4"): "full (37/37)",
    ("Q4_K_M", "f16", "T4"): "full (37/37)",
    ("Q8_0", "f16", "T4"): "full (37/37)",
}

N_PAPERS = 69


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def card_class(gib: float) -> int:
    """Smallest commodity card that holds this, with no headroom for a desktop."""
    for size in (4, 8, 12, 16, 24):
        if gib < size * 0.95:
            return size
    return 32


def derive_timings(logdir: str) -> dict:
    """Re-derive mean per-paper seconds from the retained extract logs, newest last."""
    out = {}
    for path in sorted(glob.glob(os.path.join(logdir, "extract_*.log")),
                       key=os.path.getmtime):
        text = open(path, errors="ignore").read()
        secs = [float(x) for x in re.findall(r"([0-9]+\.[0-9])s$", text, re.M)][:N_PAPERS]
        if secs:
            out[os.path.basename(path)] = sum(secs) / len(secs)
    return out


def main() -> None:
    if len(sys.argv) > 1:
        print("  timings re-derived from", sys.argv[1])
        for name, mean in derive_timings(sys.argv[1]).items():
            print(f"     {name:<34} mean {mean:5.1f}s/paper")
        print()

    print(f"  {'weights':<8} {'KV':<5} {'host':<7} {'minVRAM':>8} {'card':>5} "
          f"{'s/paper':>8} {'corpus':>8}  residency")
    print("  " + "-" * 100)
    for key in SECONDS_PER_PAPER:
        w, kv, host = key
        need = (WEIGHTS_MIB[w] + KV_MIB_AT_28672[kv] + COMPUTE_MIB) / MIB_PER_GIB
        sec = SECONDS_PER_PAPER[key]
        print(f"  {w:<8} {kv:<5} {host:<7} {need:7.2f}G {card_class(need):>4}G "
              f"{sec:7.1f}s {sec * N_PAPERS / 60:6.1f}m  {RESIDENCY[key]}")

    print()
    print("  minVRAM = measured weight buffer + KV cache at the corpus's WIDEST window")
    print("            (28,672 tokens) + compute buffer. Median papers need less; sizing")
    print("            to the median forces partial residency on the three widest.")
    print()
    print("  Accuracy is not tabulated here; `make values` recomputes it from the expert")
    print("  labels rather than quoting a stored constant.")
    print()
    best = ("Q4_K_M", "f16", "T4")
    bw = (WEIGHTS_MIB[best[0]] + KV_MIB_AT_28672[best[1]] + COMPUTE_MIB) / MIB_PER_GIB
    alt = ("Q8_0", "f16", "T4")
    aw = (WEIGHTS_MIB[alt[0]] + KV_MIB_AT_28672[alt[1]] + COMPUTE_MIB) / MIB_PER_GIB
    print(f"  OPTIMUM: {best[0]} weights + {best[1]} KV, {bw:.2f} GiB -> "
          f"{card_class(bw)} GB card")
    print(f"    vs Q8_0+f16 : +{aw - bw:.2f} GiB VRAM, "
          f"+{100 * (SECONDS_PER_PAPER[alt] / SECONDS_PER_PAPER[best] - 1):.0f}% time, "
          "-> more VRAM and more time")
    q4 = ("Q4_K_M", "q4_0", "T4")
    qw = (WEIGHTS_MIB[q4[0]] + KV_MIB_AT_28672[q4[1]] + COMPUTE_MIB) / MIB_PER_GIB
    print(f"    vs Q4_K_M+q4_0: -{bw - qw:.2f} GiB VRAM, "
          f"+{100 * (SECONDS_PER_PAPER[q4] / SECONDS_PER_PAPER[best] - 1):.0f}% time, "
          "-> the cheaper option if f16 will not fit")


if __name__ == "__main__":
    main()
