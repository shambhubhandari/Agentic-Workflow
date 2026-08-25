"""Tier 0: Deterministic schema-constrained extraction.

Executes fixed-prompt, grammar-constrained decoding across single documents. 
Bypasses dynamic agentic formulation to ensure strictly measurable precision and 
recall against the hand-labeled baseline. Missed parameters default to `reported=false`, 
meaning extraction failures strictly upper-bound true omission rates.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from .. import llm
from ..settings import INTERIM, PROMPTS, effective as _eff
from ..types import Extraction, ExtractionStatus, Paper
from .evaluate import grounded_in, value_grounded_in
from .fetch import _text_path

log = logging.getLogger(__name__)

PROMPT_ID = _eff('models', 'prompts.extract', None, "extract_method_v2")
MAX_CHARS = int(_eff('corpus', 'extract.max_chars', None, 120_000))

_REFS_HEADING = re.compile(
    r"^\s{0,6}(?:\d+\.?\s*)?(references?|bibliography|references\s+and\s+notes)\s*:?\s*$",
    re.I | re.M,
)

def _prompt() -> str:
    """Loads the fixed prompt template for the extraction task."""
    return (PROMPTS / f"{PROMPT_ID}.md").read_text(encoding="utf-8")


def _strip_references(text: str) -> str:
    """Truncates document post-bibliography to preserve prompt context."""
    match = next((m for m in reversed(list(_REFS_HEADING.finditer(text)))
                  if m.start() > len(text) * 0.45), None)
    return text[:match.start()] if match else text


# =============================================================================
#                   ********* GROUNDING VALIDATORS *********                 
#   Filters to override hallucinated extraction claims against source text.  
# =============================================================================

def _gate_ungrounded_method_fields(result: Extraction, text: str, paper_key: str) -> Extraction:
    """Reverts reported fields to False if the quoted evidence is not in the text."""
    ungrounded = []
    for name in type(result.method).model_fields:
        entry = getattr(result.method, name)
        if entry.reported and entry.evidence and not grounded_in(entry.evidence, text):
            ungrounded.append(name)
            entry.reported = False
            entry.value = None
    if ungrounded:
        log.warning("%s: %d reported fields have absent evidence, reset to not-reported: %s",
                    paper_key, len(ungrounded), ", ".join(sorted(ungrounded)))
    return result


def _gate_hallucinated_claims(result: Extraction, text: str, paper_key: str) -> Extraction:
    """Clears claim values if the exact number string does not appear in the text."""
    hallucinated = []
    for claim in (result.claims or []):
        if claim.value is None:
            continue
        try:
            val = float(claim.value)
        except ValueError:
            continue
        if not value_grounded_in(val, text):
            hallucinated.append(f"{claim.property}={claim.value}({claim.material_formula or '?'})")
            claim.value = None
    if hallucinated:
        log.warning("%s: %d claimed values appear nowhere in the paper, cleared: %s",
                    paper_key, len(hallucinated), ", ".join(hallucinated))
    return result


def _enforce_computational_status(result: Extraction, paper_key: str) -> Extraction:
    """Forces is_computational to True if any method parameters are reported."""
    n_reported = sum(
        1 for f in type(result.method).model_fields
        if getattr(result.method, f).reported
    )
    if n_reported and not result.is_computational:
        log.warning("%s: is_computational=False but %d parameters reported; overriding",
                    paper_key, n_reported)
        result.is_computational = True
    return result


# =============================================================================
#                   ********* PIPELINE EXECUTION *********                   
#      Multi-pass orchestration and deterministic LLM query dispatches.      
# =============================================================================

def extract_one(paper: Paper, model: Optional[str] = None, provider: Optional[str] = None) -> Extraction:
    """Extracts method parameters for a single paper via constrained decoding."""
    text_file = _text_path(paper.key)
    if not text_file.exists():
        return Extraction(
            paper_key=paper.key, doi=paper.doi,
            status=ExtractionStatus.NO_FULLTEXT, error="no cached full text",
        )

    text = _strip_references(text_file.read_text(encoding="utf-8"))[:MAX_CHARS]
    model_used = model or llm.agent_model(provider, "extract")[1]

    try:
        result, model_used = llm.structured(
            f"{_prompt()}\n\n---\n\n{text}",
            Extraction, agent="extract", provider=provider, model=model,
            require_all=False,
            require_fields={
                "reported", "is_computational", "is_pentagonal_2d",
                "claims", "value", "unit", "material_formula",
            },
        )
    except Exception as exc:
        return Extraction(
            paper_key=paper.key, doi=paper.doi,
            status=ExtractionStatus.FAILED, error=f"{type(exc).__name__}: {exc}",
            model=model_used, prompt_id=PROMPT_ID,
        )

    result = _gate_ungrounded_method_fields(result, text, paper.key)
    result = _gate_hallucinated_claims(result, text, paper.key)
    result = _enforce_computational_status(result, paper.key)

    result.paper_key = paper.key
    result.doi = paper.doi
    result.model = model_used
    result.prompt_id = PROMPT_ID

    if result.status == ExtractionStatus.OK and not result.is_computational:
        result.status = ExtractionStatus.NOT_COMPUTATIONAL

    return result


# =============================================================================
#                       ********* PUBLIC API *********                       
#         Entrypoints for bulk corpus extraction and state recovery.         
# =============================================================================

def extract_passes(papers: list[Paper], n_passes: Optional[int] = None,
                   limit: Optional[int] = None, force: bool = False) -> dict[str, int]:
    """Runs extraction N independent times, then combines by grounded union."""
    from . import consensus

    n_passes = int(n_passes or _eff('corpus', 'extract.passes', None, 1))
    pass_dir = INTERIM / "passes"
    pass_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(1, n_passes + 1):
        path = pass_dir / f"extracted.pass{i}.jsonl"
        if path.exists() and not force:
            log.info("pass %d/%d: reusing %s", i, n_passes, path.name)
        else:
            log.info("pass %d/%d -> %s", i, n_passes, path.name)
            extract_all(papers, limit=limit, out_path=path, force=True)
        paths.append(path)

    stats = consensus.combine(paths)
    stats["n_passes_requested"] = n_passes
    return stats


def extract_all(papers: list[Paper], limit: Optional[int] = None,
                out_path: Optional[Path] = None, force: bool = False) -> list[Extraction]:
    """Extracts every paper with cached full text, resumable via output JSONL."""
    out_path = out_path or (INTERIM / "extracted.jsonl")

    done: dict[str, Extraction] = {}
    if out_path.exists() and not force:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = Extraction(**json.loads(line))
                done[rec.paper_key] = rec

    limit = _eff('corpus', 'extract.limit', limit)
    targets = [p for p in papers if _text_path(p.key).exists()]
    if limit:
        targets = targets[:limit]

    results: list[Extraction] = []
    mode = "w" if force else "a"
    
    with open(out_path, mode, encoding="utf-8") as fh:
        for i, paper in enumerate(targets, 1):
            if paper.key in done:
                results.append(done[paper.key])
                continue
            
            t0 = time.time()
            rec = extract_one(paper)
            fh.write(rec.model_dump_json() + "\n")
            fh.flush()
            results.append(rec)
            
            n_rep = sum(
                1 for f in type(rec.method).model_fields
                if getattr(rec.method, f).reported
            )
            log.info("[%3d/%d] %-34s %-18s %2d reported  %.1fs",
                     i, len(targets), paper.key[:34], rec.status.value, n_rep, time.time() - t0)

    return results


def load(path: Optional[Path] = None) -> list[Extraction]:
    """Loads a collection of Extraction records from a JSONL file."""
    path = path or (INTERIM / "extracted.jsonl")
    if not Path(path).exists():
        published = INTERIM / "extraction" / "rtx3050_q4_0" / "union.jsonl"
        if published.exists():
            path = published
    with open(path, encoding="utf-8") as fh:
        return [Extraction(**json.loads(line)) for line in fh if line.strip()]
