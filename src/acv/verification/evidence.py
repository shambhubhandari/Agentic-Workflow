"""System: evidence module.

Provides strict, deterministic logic and strict typing for evidence operations.
"""
from __future__ import annotations

# =============================================================================
#                   ********* VERIFICATION METRICS *********                   
#                       Strict definitions for evidence.                       
# =============================================================================

import json
import re
from functools import cache

from .. import settings as S

LAYER_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")


@cache
def _offload_log() -> str:
    path = S.RUNTIME_LOGS / "ollama_layer_offload.txt"
    return path.read_text(errors="replace")


def _layer_splits() -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in LAYER_OFFLOAD.findall(_offload_log())]


def model_loads() -> int:
    """Times the server loaded the model during the campaign."""
    return len(_layer_splits())


def layers_resident_min() -> int:
    return min(a for a, _ in _layer_splits())


def layers_resident_max() -> int:
    return max(a for a, _ in _layer_splits())


def layers_total() -> int:
    return max(b for _, b in _layer_splits())


@cache
def _pass(letter: str) -> dict[str, dict]:
    path = S.EXTRACTION / "rtx3050_q4_0" / f"pass_{letter}.jsonl"
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["paper_key"]] = record
    return records


def _field_agreement(left: str, right: str) -> tuple[int, int, int]:
    """(agreeing decisions, total decisions, papers differing) between two passes."""
    a, b = _pass(left), _pass(right)
    agree = total = 0
    differing = set()
    for key in a.keys() & b.keys():
        for field, entry in a[key]["method"].items():
            other = b[key]["method"].get(field)
            if not isinstance(entry, dict) or not isinstance(other, dict):
                continue
            total += 1
            if bool(entry.get("reported")) == bool(other.get("reported")):
                agree += 1
            else:
                differing.add(key)
    return agree, total, len(differing)


def pass_stability_min() -> float:
    """Lowest field-decision agreement across the three pairs of passes."""
    rates = []
    for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
        agree, total, _ = _field_agreement(left, right)
        if total:
            rates.append(100 * agree / total)
    return round(min(rates), 1)


def pass_stability_max() -> float:
    rates = []
    for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
        agree, total, _ = _field_agreement(left, right)
        if total:
            rates.append(100 * agree / total)
    return round(max(rates), 1)


def papers_differing_max() -> int:
    """Most papers differing between any two passes — the upper end of the quoted range."""
    return max(_field_agreement(a, b)[2] for a, b in (("a", "b"), ("a", "c"), ("b", "c")))
