"""System: corpus module.

Provides strict, deterministic logic and strict typing for corpus operations.
"""
from __future__ import annotations

# =============================================================================
#                   ********* VERIFICATION METRICS *********                   
#                        Strict definitions for corpus.                        
# =============================================================================

import json
from functools import cache

from scipy.stats import fisher_exact

from .. import settings as S

ARXIV_PREFIX = "10.48550"


@cache
def _corpus() -> tuple[dict, ...]:
    text = S.CORPUS_LOCKED.read_text(encoding="utf-8")
    return tuple(json.loads(line) for line in text.splitlines() if line.strip())


@cache
def _reportability() -> tuple[dict, ...]:
    text = S.REPORTABILITY.read_text(encoding="utf-8")
    return tuple(json.loads(line) for line in text.splitlines() if line.strip())


@cache
def _summary() -> dict:
    return json.loads(S.EXTRACTION_SUMMARY.read_text(encoding="utf-8"))


def corpus_records() -> int:
    """Papers in the frozen OpenAlex query the study used."""
    return len(_corpus())


def open_access_records() -> int:
    return sum(1 for r in _corpus() if r.get("oa_url"))


def fulltext_retrieved() -> int:
    """Documents whose text was retrieved; the text itself is not redistributed."""
    lines = S.FULLTEXT_MANIFEST.read_text(encoding="utf-8").splitlines()
    return max(len(lines) - 1, 0)


def papers_extracted() -> int:
    return len(_reportability())


def papers_audited() -> int:
    """Extracted papers minus those the extractor judged non-pentagonal on full text."""
    return _summary()["n_usable"]


def papers_fully_reporting() -> int:
    return _summary()["reproducible_in_principle"]


def mean_reporting_rate() -> float:
    return round(100 * _summary()["mean_fraction_reported"], 1)


@cache
def _excluded_keys() -> frozenset[str]:
    """Papers the extractor judged non-pentagonal on reading the full text.

    Recorded as is_pentagonal_2d=False in the extraction, not in the reportability file,
    Enforce cross-join to recover audited population.
    """
    text = S.union("rtx3050_q4_0").read_text(encoding="utf-8")
    excluded = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("is_pentagonal_2d") is False:
            excluded.add(record["paper_key"])
    return frozenset(excluded)


def papers_excluded_not_pentagonal() -> int:
    """Audited-set exclusions.

    Ten extraction records carry is_pentagonal_2d=False, but only those that also reached
    the reportability stage reduce the audited population; the rest failed earlier.
    """
    keys = {r["paper_key"] for r in _reportability()}
    return len(_excluded_keys() & keys)


def _split_by_publication() -> tuple[list, list]:
    """Audited papers split by publication route, arXiv-only against publisher DOI."""
    rows = [r for r in _reportability() if r["paper_key"] not in _excluded_keys()]
    return ([r for r in rows if _is_arxiv(r)],
            [r for r in rows if not _is_arxiv(r)])


def _is_arxiv(row: dict) -> bool:
    return str(row.get("doi") or "").lower().startswith(ARXIV_PREFIX)


def preprint_papers() -> int:
    return len(_split_by_publication()[0])


def preprint_fully_reporting() -> int:
    return sum(1 for r in _split_by_publication()[0] if r.get("reproducible_in_principle"))


def publisher_fully_reporting() -> int:
    return sum(1 for r in _split_by_publication()[1] if r.get("reproducible_in_principle"))


def preprint_publisher_fisher() -> float:
    """Two-sided Fisher exact on completeness by publication route."""
    pre, pub = _split_by_publication()
    a = sum(1 for r in pre if r.get("reproducible_in_principle"))
    b = sum(1 for r in pub if r.get("reproducible_in_principle"))
    table = [[a, len(pre) - a], [b, len(pub) - b]]
    return round(float(fisher_exact(table, alternative="two-sided")[1]), 3)
