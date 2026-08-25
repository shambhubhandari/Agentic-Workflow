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
    text = S.PARITY_POINTS.read_text(encoding="utf-8")
    return tuple(json.loads(line) for line in text.splitlines() if line.strip())


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
