"""Tier 0: Corpus construction and determinism enforcement.

Executes paginated queries against the OpenAlex API, filters mathematically
impossible compositions via stoichiometry (`penta.detect`), and builds 
a deduplicated, hash-sampled target population.
"""

from __future__ import annotations

import collections
import json
import logging
import time
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import requests

from ..settings import INTERIM, OPENALEX_BASE, OPENALEX_MAILTO, RAW
from ..settings import effective as _eff
from ..types import Paper
from . import penta

log = logging.getLogger(__name__)

# =============================================================================
#                     ********* CONFIGURATION *********                      
#         Seed terms and query constraints for corpus construction.          
# =============================================================================

SEED_TERMS: list[str] = _eff('corpus', 'search.seed_terms', None, [
    "penta-graphene", "penta-monolayer", "pentagonal monolayer", "pentagonal nanosheet",
    "pentagonal lattice", "pentagonal two-dimensional", "Cairo pentagonal",
    "Cairo tiling", "Cairo lattice", "penta-SiC2", "penta-CN2", "penta-BN2",
    "penta-BCN", "penta-SiCN", "penta-PdSe2", "penta-NiN2",
])

COMPUTATIONAL_HINTS: tuple[str, ...] = (
    "first-principles", "first principles", "density functional", "dft",
    "ab initio", "vasp", "siesta", "quantum espresso", "castep",
)

ALLOWED_TYPES: set[str] = {"article", "preprint"}
_SUPPLEMENTARY_DOI = re.compile(r"\.s\d{3}$|/suppl")

REQUEST_DELAY_S: float = 0.4
MAX_RETRY_WAIT_S: int = int(_eff('corpus', 'search.max_retry_wait_s', None, 120))
DEFAULT_FROM_YEAR: Optional[int] = _eff('corpus', 'search.from_year', None, None)
DEFAULT_REQUIRE_COMPUTATIONAL: bool = bool(_eff('corpus', 'search.require_computational', None, True))
DEFAULT_MAX_PAPERS: Optional[int] = _eff('corpus', 'search.max_papers', None, None)

# =============================================================================
#                      ********* OPENALEX API *********                      
#      Network procurement and strict geometric/computational filtering.     
# =============================================================================

class BudgetExhausted(RuntimeError):
    pass


def _get(url: str, params: dict[str, Any], retries: int = 7) -> dict[str, Any]:
    """GET with exponential backoff, honouring Retry-After when OpenAlex sends it."""
    params = {**params, "mailto": OPENALEX_MAILTO}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                time.sleep(REQUEST_DELAY_S)
                return r.json()
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", 0))
                remaining = r.headers.get("x-ratelimit-remaining-usd")
                if retry_after > MAX_RETRY_WAIT_S:
                    raise BudgetExhausted(
                        f"OpenAlex daily budget exhausted (remaining ${remaining}). "
                        f"Resets in {retry_after / 3600:.1f}h at midnight UTC."
                    )
                wait = retry_after or 2**attempt
                log.warning("OpenAlex 429 (attempt %d/%d); waiting %.0fs", attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                wait = min(2**attempt, 60)
                log.warning("OpenAlex %s (attempt %d/%d); waiting %.0fs", r.status_code, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.RequestException as exc:
            wait = min(2**attempt, 60)
            log.warning("OpenAlex request failed (%s); retrying in %.0fs", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"OpenAlex request failed after {retries} attempts: {url}")


def _looks_computational(work: dict[str, Any]) -> bool:
    """True if the abstract or title contains recognizable DFT terminology."""
    text = f"{work.get('title') or ''} {work.get('abstract_inverted_index') or ''}".lower()
    return any(hint in text for hint in COMPUTATIONAL_HINTS)


def _reconstruct_abstract(inverted: dict[str, list[int]]) -> str:
    """Rebuilds OpenAlex's inverted index into a readable string."""
    if not inverted:
        return ""
    length = max((idx for positions in inverted.values() for idx in positions), default=-1) + 1
    words = [""] * length
    for word, positions in inverted.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def search_term(term: str, from_year: Optional[int] = None) -> Iterator[dict[str, Any]]:
    """Yields all OpenAlex works matching the query phrase."""
    filters = [f"title_and_abstract.search:{term}"]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
        
    cursor = "*"
    while cursor:
        page = _get(
            f"{OPENALEX_BASE}/works",
            {"filter": ",".join(filters), "per-page": 200, "cursor": cursor},
        )
        for work in page.get("results", []):
            yield work
        cursor = page.get("meta", {}).get("next_cursor")


def to_paper(work: dict[str, Any], term: str) -> Optional[Paper]:
    """Validates the work type and stoichiometric plausibility before instantiating."""
    if work.get("type") not in ALLOWED_TYPES:
        return None
        
    doi = (work.get("doi") or "").replace("https://doi.org/", "").strip()
    if not doi:
        return None
    if _SUPPLEMENTARY_DOI.search(doi.lower()):
        return None

    title = (work.get("title") or "").strip()
    if not title:
        return None

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or {})
    text = f"{title} {abstract}"
    
    if "penta" not in text.lower():
        return None

    compositions = penta.detect(text)
    if not compositions:
        return None

    year = work.get("publication_year")
    key = doi if doi else work["id"].split("/")[-1]

    return Paper(
        key=key,
        doi=doi,
        year=year,
        title=title,
        matched_terms=[term],
        compositions=list(compositions),
        is_open_access=bool(work.get("open_access", {}).get("is_oa")),
        oa_url=work.get("open_access", {}).get("oa_url"),
    )


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Resolves duplicated entries by preferring the DOIs from the primary publisher."""
    by_doi: dict[str, Paper] = {}
    for p in papers:
        if p.doi:
            existing = by_doi.get(p.doi)
            if existing:
                existing.matched_terms = list(set(existing.matched_terms + p.matched_terms))
                existing.compositions = list(set(existing.compositions + p.compositions))
                if p.is_open_access and not existing.is_open_access:
                    existing.is_open_access = True
                    existing.oa_url = p.oa_url
            else:
                by_doi[p.doi] = p
                
    out = list(by_doi.values())
    for p in papers:
        if not p.doi:
            out.append(p)
    return out


# =============================================================================
#                     ********* ORCHESTRATION *********                      
#         Checkpointing, deterministic sampling, and state recovery.         
# =============================================================================

def build(
    terms: Optional[list[str]] = None,
    from_year: Optional[int] = None,
    require_computational: Optional[bool] = None,
    out_path: Optional[Path] = None,
    max_papers: Optional[int] = DEFAULT_MAX_PAPERS,
) -> list[Paper]:
    """Constructs the corpus and strictly writes to JSONL."""
    terms = terms or SEED_TERMS
    from_year = DEFAULT_FROM_YEAR if from_year is None else from_year
    if require_computational is None:
        require_computational = DEFAULT_REQUIRE_COMPUTATIONAL
    out_path = Path(out_path or (RAW / "corpus.jsonl"))

    papers: dict[str, Paper] = {}
    cache_dir = INTERIM / "corpus_terms"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for term in terms:
        slug = term.replace(" ", "_").replace("/", "_")
        cached = cache_dir / f"{slug}.jsonl"

        if cached.exists():
            found = [Paper(**json.loads(line))
                     for line in cached.read_text().splitlines() if line.strip()]
            log.info("%-28s %4d cached", term, len(found))
        else:
            found = []
            try:
                for work in search_term(term, from_year=from_year):
                    if require_computational and not _looks_computational(work):
                        continue
                    paper = to_paper(work, term)
                    if paper is not None:
                        found.append(paper)
            except BudgetExhausted as exc:
                log.error("%s", exc)
                log.error("Stopping. %d/%d terms cached; re-run to resume.",
                          len(list(cache_dir.glob("*.jsonl"))), len(terms))
                break
            except RuntimeError as exc:
                log.error("%-28s FAILED (%s) - will retry on next run", term, exc)
                continue
                
            cached.write_text("\n".join(p.model_dump_json() for p in found) + "\n")
            log.info("%-28s %4d kept", term, len(found))

        for paper in found:
            existing = papers.get(paper.key)
            if existing:
                for t in paper.matched_terms:
                    if t not in existing.matched_terms:
                        existing.matched_terms.append(t)
            else:
                papers[paper.key] = paper

    ordered = deduplicate(list(papers.values()))
    ordered = sorted(ordered, key=lambda p: (-(p.year or 0), p.key))

    if max_papers is not None and max_papers <= 0:
        max_papers = None

    if max_papers is not None and len(ordered) > max_papers:
        import hashlib
        before = collections.Counter(p.year for p in ordered)
        chosen = sorted(ordered, key=lambda p: hashlib.sha256(p.key.encode()).hexdigest())
        ordered = sorted(chosen[:max_papers], key=lambda p: (-(p.year or 0), p.key))
        after = collections.Counter(p.year for p in ordered)
        log.warning("SAMPLED %d of %d papers (deterministic, hash-ordered).", max_papers, len(chosen))
        years = sorted(set(before) | set(after), key=lambda y: -(y or 0))
        log.info("year spread kept: %s", ", ".join(f"{y}:{after.get(y,0)}/{before.get(y,0)}" for y in years if y))

    if not ordered:
        raise RuntimeError(f"Refusing to write an empty corpus to {out_path}.")

    if out_path.exists():
        previous = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if previous and len(ordered) < previous * 0.5:
            backup = out_path.with_suffix(".jsonl.bak")
            out_path.replace(backup)
            log.warning("New corpus (%d) is < half the previous (%d); saved backup to %s",
                        len(ordered), previous, backup.name)

    with open(out_path, "w", encoding="utf-8") as fh:
        for paper in ordered:
            fh.write(paper.model_dump_json() + "\n")

    log.info("corpus: %d unique papers -> %s", len(ordered), out_path)
    return ordered


def load(path: Optional[Path] = None) -> list[Paper]:
    """Read a previously built corpus."""
    path = Path(path or (RAW / "corpus.jsonl"))
    with open(path, encoding="utf-8") as fh:
        return [Paper(**json.loads(line)) for line in fh if line.strip()]
