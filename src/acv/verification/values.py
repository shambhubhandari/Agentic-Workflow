"""System: values module.

Provides strict, deterministic logic and strict typing for values operations.
"""
from __future__ import annotations

# =============================================================================
#                   ********* VERIFICATION METRICS *********                   
#                        Strict definitions for values.                        
# =============================================================================

import json
from collections import Counter
from functools import cache

from .. import settings as S
from . import metrics

PUBLISHED = "rtx3050_q4_0"          # the configuration the manuscript reports


@cache
def _labels() -> tuple[dict, ...]:
    return tuple(metrics.load_labels())


@cache
def _counts(configuration: str = PUBLISHED) -> Counter:
    return metrics.confusion(list(_labels()),
                             metrics.load_extraction(S.union(configuration)))


@cache
def _by_configuration() -> dict:
    return metrics.configuration_metrics()


@cache
def _parity() -> tuple[dict, ...]:
    """The comparisons every Tier 2 statistic is computed over.

    parity_points.jsonl also carries the targets the Critic held, so the figure can show
    them faded for disclosure. They are dropped here: a relaxation that left its
    prototype measures the starting geometry, so counting it would report entrapment as
    if it were reproduction error.
    """
    text = S.PARITY_POINTS.read_text(encoding="utf-8")
    rows = (json.loads(line) for line in text.splitlines() if line.strip())
    return tuple(r for r in rows if not r.get("held"))


# ------------------------------------------------------------------ published run

def precision_presence() -> float:
    return round(metrics.presence(_counts())[0], 1)


def recall_presence() -> float:
    return round(metrics.presence(_counts())[1], 1)


def f1_presence() -> float:
    return round(metrics.presence(_counts())[2], 1)


def recall_ci_low() -> float:
    return round(_recall_interval()[0], 1)


def recall_ci_high() -> float:
    return round(_recall_interval()[1], 1)


def _recall_interval() -> tuple[float, float]:
    counts = _counts()
    return metrics.wilson(counts["tp"], counts["tp"] + counts["fn"])


def false_positives() -> int:
    return _counts()["fp"]


def judgements_scored() -> int:
    return sum(_counts().values())


def papers_labelled() -> int:
    """Papers contributing at least one scorable judgement; all-n/a stubs contribute none."""
    return len({row["paper_key"] for row in _labels()
                if (row.get("reported") or "").strip().lower() in ("y", "n")})


def recall_eight_fields() -> float:
    """Recall over the fields every code family shares; basis size applies only to NAO codes."""
    counts = metrics.confusion(list(_labels()),
                               metrics.load_extraction(S.union(PUBLISHED)),
                               fields=metrics.FIELDS_ALL_CODE_FAMILIES)
    return round(metrics.presence(counts)[1], 1)


def basis_size_positives() -> int:
    per_field = metrics.per_field_recall(list(_labels()),
                                         metrics.load_extraction(S.union(PUBLISHED)))
    return per_field["basis_size"][1]


def constant_baseline_f1() -> float:
    return round(metrics.constant_baseline_f1(list(_labels())), 1)


# ------------------------------------------------------- across configurations

def _statistic(configuration: str, orientation: str, name: str) -> float:
    return round(_by_configuration()[configuration][orientation][name], 1)


def _mcc(configuration: str) -> float:
    return round(_by_configuration()[configuration]["mcc"], 3)


def rtx_mcc() -> float:
    return _mcc("rtx3050_q4_0")


def t4_q4_mcc() -> float:
    return _mcc("tesla_t4_q4_0")


def t4_f16_mcc() -> float:
    return _mcc("tesla_t4_f16")


def t4_f16_precision() -> float:
    return _statistic("tesla_t4_f16", "presence", "precision")


def t4_q4_recall() -> float:
    return _statistic("tesla_t4_q4_0", "presence", "recall")


def precision_absence_best() -> float:
    """Best absence precision across configurations — the bound the abstract quotes."""
    return round(max(v["absence"]["precision"] for v in _by_configuration().values()), 1)


# --------------------------------------------------------- Tier 2 recomputation

def parity_comparisons() -> int:
    return len(_parity())


def parity_mae() -> float:
    points = _parity()
    return round(sum(abs(p["ours"] - p["claimed"]) for p in points) / len(points), 3)


def parity_mare() -> float:
    points = _parity()
    relative = (100 * abs(p["ours"] - p["claimed"]) / p["claimed"] for p in points)
    return round(sum(relative) / len(points), 1)


def _parity_relative() -> list[float]:
    return sorted(100 * abs(p["ours"] - p["claimed"]) / p["claimed"] for p in _parity())


def parity_prototypes() -> int:
    return len({p["formula"] for p in _parity()})


def parity_median() -> float:
    """Median |relative deviation|, the robust counterpart to the MARE."""
    r = _parity_relative()
    mid, odd = len(r) // 2, len(r) % 2
    return round(r[mid] if odd else (r[mid - 1] + r[mid]) / 2, 1)


def parity_bias() -> float:
    """Signed mean deviation: whether the recomputation is centred or offset."""
    points = _parity()
    return round(sum(p["ours"] - p["claimed"] for p in points) / len(points), 3)


def parity_within_two() -> int:
    return sum(1 for r in _parity_relative() if r <= 2.0)


# ------------------------------------------------------- multi-pass union (Table 2)

@cache
def _passes() -> tuple[list[int], list[int], int, int]:
    """(fields per pass, papers reporting none per pass, union fields, union zeros)."""
    base = S.INTERIM / "extraction" / PUBLISHED

    def load(name):
        return [json.loads(x) for x in (base / name).read_text(encoding="utf-8").splitlines()
                if x.strip()]

    def reported(rec):
        return sum(1 for e in rec["method"].values()
                   if isinstance(e, dict) and e.get("reported"))

    passes = [load(f"pass_{k}.jsonl") for k in "abc"]
    union = load("union_gated.jsonl")
    return ([sum(reported(r) for r in p) for p in passes],
            [sum(1 for r in p if not reported(r)) for p in passes],
            sum(reported(r) for r in union),
            sum(1 for r in union if not reported(r)))


def pass_fields_one() -> int:
    return _passes()[0][0]


def pass_fields_two() -> int:
    return _passes()[0][1]


def pass_fields_three() -> int:
    return _passes()[0][2]


def union_recovered_over_best() -> int:
    """Fields the union adds over the best single pass, after collision resolution."""
    fields, _, union, _ = _passes()
    return union - max(fields)


# --------------------------------------- cache precision against computed physics

@cache
def _cache_pairs() -> list[tuple[float, float]]:
    """Recomputed cells for every target run under both T4 cache precisions.

    Compared cell-to-cell rather than through the parity plot: the question is whether
    KV-cache precision changes the electronic structure, which is a recomputed-against-
    recomputed comparison and has nothing to do with what a paper reported. Targets the
    Critic held are kept -- a held structure is still valid evidence that two precisions
    produce the same cell.
    """
    q4 = {p.name: json.loads(p.read_text(encoding="utf-8"))
          for p in (S.PROCESSED / "verification_tesla_t4_q4_0").glob("*.json")}
    f16 = {p.name: json.loads(p.read_text(encoding="utf-8"))
           for p in (S.PROCESSED / "verification_tesla_t4_f16").glob("*.json")}
    out = []
    for name in sorted(set(q4) & set(f16)):
        a, b = q4[name], f16[name]
        cells = (a.get("our_a"), a.get("our_b"), b.get("our_a"), b.get("our_b"))
        if None in cells:                       # a collapsed run has no cell to compare
            continue
        out.append((abs(cells[0] - cells[2]), abs(cells[1] - cells[3])))
    return out


def cache_shared_targets() -> int:
    return len(_cache_pairs())


def cache_cell_agreement() -> float:
    """Largest in-plane cell difference between the two cache precisions, in angstrom."""
    return round(max(max(p) for p in _cache_pairs()), 4)
