"""Tier 0.2: Network retrieval and caching of unstructured full text.

Executes polite HTTP retrieval of open-access manuscripts from arXiv and 
publisher endpoints. Extracts plaintext via PyMuPDF/BeautifulSoup and 
validates semantic completeness via vocabulary heuristics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from ..settings import FULLTEXT
from ..settings import effective as _eff
from ..types import Paper

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & SETTINGS *********                  
#         Network delays, user agents, and semantic gating thresholds.         
# =============================================================================

ARXIV_DELAY_S: float = float(_eff('corpus', 'fetch.arxiv_delay_s', None, 3.0))
OA_DELAY_S: float = float(_eff('corpus', 'fetch.oa_delay_s', None, 1.0))

USER_AGENT: str = (
    "ACV/0.1 (reproducibility audit of pentagonal 2D materials literature; "
    "mailto:acv-research@ucl.ac.uk)"
)

MIN_USEFUL_CHARS: int = int(_eff('corpus', 'fetch.min_useful_chars', None, 2000))

ARTICLE_MARKERS: tuple[str, ...] = (
    "abstract", "introduction", "method", "calculat", "result",
    "conclusion", "references", "figure", "et al",
)
MIN_MARKERS: int = int(_eff('corpus', 'fetch.min_article_markers', None, 4))

METHOD_MARKERS: tuple[str, ...] = (
    "cutoff", "cut-off", "k-point", "k point", "kpoint", "brillouin",
    "convergence", "functional", "pseudopotential", "relax", "supercell",
)
MIN_METHOD_MARKERS: int = int(_eff('corpus', 'fetch.min_method_markers', None, 3))
MIN_BODY_CHARS: int = int(_eff('corpus', 'fetch.min_body_chars', None, 12_000))


@dataclass
class FetchResult:
    key: str
    ok: bool
    source: Optional[str] = None
    n_chars: int = 0
    error: Optional[str] = None


# =============================================================================
#                     ********* PARSING BACKENDS *********                   
#           Binary translation from PDF/HTML into unstructured plaintext.      
# =============================================================================

def _text_path(key: str) -> Path:
    return FULLTEXT / f"{key}.txt"


def _meta_path(key: str) -> Path:
    return FULLTEXT / f"{key}.meta.json"


def _pdf_to_text(data: bytes) -> str:
    """Extract text from PDF bytes with PyMuPDF."""
    import fitz
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _html_to_text(data: bytes) -> str:
    """Extract text from HTML bytes using BeautifulSoup."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


# =============================================================================
#                    ********* NETWORK ORCHESTRATION *********               
#         Stateful retrieval, caching, and document validity gating.         
# =============================================================================

def _download(url: str, timeout: int = 60) -> tuple[bytes, str]:
    """Return (content, content_type). Raises on non-200."""
    r = requests.get(
        url, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
    )
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")


def fetch_one(paper: Paper, force: bool = False) -> FetchResult:
    """Fetch full text for one paper, preferring arXiv over the publisher's OA copy."""
    key = paper.key
    text_file = _text_path(key)

    if text_file.exists() and not force:
        return FetchResult(key, True, "cache", len(text_file.read_text(encoding="utf-8")))

    attempts: list[tuple[str, str, float]] = []
    if paper.arxiv_id:
        attempts.append(("arxiv", f"https://arxiv.org/pdf/{paper.arxiv_id}", ARXIV_DELAY_S))

    for url in [*(paper.oa_locations or []), paper.oa_url]:
        if not url or any(url == u for _, u, _ in attempts):
            continue
        kind = "oa_pdf" if url.lower().endswith(".pdf") else "oa_html"
        attempts.append((kind, url, OA_DELAY_S))

    if not attempts:
        return FetchResult(key, False, error="no retrievable source")

    last_error = None
    for source, url, delay in attempts:
        try:
            content, ctype = _download(url)
            time.sleep(delay)

            if content[:4] == b"%PDF" or "pdf" in ctype.lower():
                text = _pdf_to_text(content)
                source = "arxiv" if source == "arxiv" else "oa_pdf"
            elif "html" in ctype.lower():
                text = _html_to_text(content)
                source = "oa_html"
            else:
                last_error = f"unhandled content-type {ctype!r}"
                continue

            if len(text) < MIN_USEFUL_CHARS:
                last_error = f"only {len(text)} chars from {source}"
                continue

            low = text.lower()
            n_markers = sum(1 for m in ARTICLE_MARKERS if m in low)
            if n_markers < MIN_MARKERS:
                last_error = f"not article-like ({n_markers} markers) from {source}"
                continue

            n_method = sum(1 for m in METHOD_MARKERS if m in low)
            if len(text) < MIN_BODY_CHARS and n_method < MIN_METHOD_MARKERS:
                last_error = (
                    f"abstract-only ({len(text)} chars, {n_method} method terms) from {source}"
                )
                continue

            text_file.write_text(text, encoding="utf-8")
            _meta_path(key).write_text(
                json.dumps(
                    {
                        "key": key,
                        "doi": paper.doi,
                        "source": source,
                        "url": url,
                        "n_chars": len(text),
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return FetchResult(key, True, source, len(text))

        except requests.RequestException as exc:
            last_error = f"{source}: {exc}"
            time.sleep(delay)
        except Exception as exc:
            last_error = f"{source}: {type(exc).__name__}: {exc}"

    return FetchResult(key, False, error=last_error)


def fetch_all(
    papers: list[Paper], limit: Optional[int] = None, force: bool = False
) -> list[FetchResult]:
    """Fetch full text for every retrievable paper."""
    from ..settings import effective
    limit = effective('corpus', 'fetch.limit', limit)
    targets = [p for p in papers if p.retrievable]
    if limit:
        targets = targets[:limit]

    results: list[FetchResult] = []
    for i, paper in enumerate(targets, 1):
        res = fetch_one(paper, force=force)
        results.append(res)
        status = res.source if res.ok else f"FAIL ({res.error})"
        log.info("[%3d/%d] %-38s %s", i, len(targets), paper.key[:38], status)

    ok = sum(1 for r in results if r.ok)
    log.info("fetched %d/%d (%d cached)", ok, len(results),
             sum(1 for r in results if r.source == "cache"))
    return results
