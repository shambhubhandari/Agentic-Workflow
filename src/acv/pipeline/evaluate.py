"""Tier 1: Grounded verification and unit-aware metric evaluation.

Executes source-text validation of extracted parameters and computes
precision, recall, and F1 scoring against manually curated baselines.
Implements unit-conversion heuristics (e.g., eV to Ry) to avoid 
manufacturing false negatives on structurally correct extractions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..settings import INTERIM, PROCESSED
from ..types import Extraction
from .fetch import _text_path

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & UNITS *********                   
#      CODATA-aligned conversion factors for extraction normalization.       
# =============================================================================

RY_TO_EV = 13.605693
RY_BOHR_TO_EV_ANG = 25.71104
NM_TO_ANG = 10.0
HARTREE_TO_EV = 27.211386

REL_TOL = 0.01

FUZZY_FLOOR = 0.85

NGRAM_WORDS = 6


@dataclass
class FieldCheck:
    paper_key: str
    field: str
    value: Optional[float | str] = None
    grounded: Optional[bool] = None
    match: str = ""
    supported: Optional[bool] = None
    via: str = ""
    evidence: str = ""


@dataclass
class EvalReport:
    n_records: int = 0
    n_with_text: int = 0
    checks: list[FieldCheck] = field(default_factory=list)

    def summary(self) -> dict:
        grounded = [c for c in self.checks if c.grounded is not None]
        supported = [c for c in self.checks if c.supported is not None]
        by_field: dict[str, dict] = {}
        for c in grounded:
            entry = by_field.setdefault(c.field, {"n": 0, "grounded": 0})
            entry["n"] += 1
            entry["grounded"] += int(bool(c.grounded))
        return {
            "n_records": self.n_records,
            "n_with_text": self.n_with_text,
            "n_reported_fields": len(grounded),
            "grounding": {
                "exact": sum(1 for c in grounded if c.match == "exact"),
                "fuzzy": sum(1 for c in grounded if c.match == "fuzzy"),
                "absent": sum(1 for c in grounded if c.match == "absent"),
                "no_evidence": sum(1 for c in grounded if c.match == "no_evidence"),
                "rate": (sum(1 for c in grounded if c.grounded) / len(grounded)
                         if grounded else None),
            },
            "value_support": {
                "supported": sum(1 for c in supported if c.supported),
                "unsupported": sum(1 for c in supported if not c.supported),
                "rate": (sum(1 for c in supported if c.supported) / len(supported)
                         if supported else None),
                "via": _counts(c.via for c in supported if c.supported and c.via),
            },
            "by_field": {
                k: {**v, "rate": v["grounded"] / v["n"]}
                for k, v in sorted(by_field.items(), key=lambda x: x[1]["n"], reverse=True)
            },
        }


def _counts(items) -> dict:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


_UNICODE_FIXES = {"\u2212": "-", "\u00d7": "x", "\u2019": "'", "\u2018": "'",
                  "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                  "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\u00a0": " "}

# =============================================================================
#                   ********* GROUNDING VALIDATORS *********                 
#       Source-text string matchers preventing fabricated parameters.        
# =============================================================================

_QUOTE_FRAME = re.compile(
    r"^\s*(?:the\s+)?(?:paper|article|authors?|text|section\s+\S+)\s+"
    r"(?:states?|says?|reports?|writes?|notes?|mentions?)\s*:?\s*['\"]?", re.I)


def _norm(text: str) -> str:
    """Collapse whitespace, case and typography so PDF artefacts are not read as
    fabrication. Ligatures and minus signs survive extraction as distinct codepoints and
    would otherwise make a verbatim quote look like a different sentence."""
    for bad, good in _UNICODE_FIXES.items():
        text = text.replace(bad, good)
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_grounded(evidence: str, haystack: str) -> str:
    """exact | fuzzy | absent.

    A genuine quote shares an uninterrupted RUN OF WORDS with the source; a fabricated or
    self-negating one does not. Whole-string similarity was tried and fails in both
    directions: `quick_ratio()` ignores word order, so "the paper does not explicitly
    state the mesh cutoff" -- written by the model, absent from the paper -- scored 0.867
    against ordinary English prose. Tightening to `ratio()` then rejected 63 real methods
    sentences, because the model frames its quotes ("The paper states: '...'"), because
    PDFs mangle typography, and because a truncated probe misaligns against a sliding
    window. Matching on the longest shared word run is invariant to all three.
    """
    needle = _QUOTE_FRAME.sub("", _norm(evidence)).strip(" '\"")
    if not needle:
        return "no_evidence"
    if needle in haystack:
        return "exact"
    words = needle.split()
    if len(words) < NGRAM_WORDS:
        return "fuzzy" if needle in haystack else "absent"
    for i in range(len(words) - NGRAM_WORDS + 1):
        if " ".join(words[i:i + NGRAM_WORDS]) in haystack:
            return "fuzzy"
    return "absent"


def grounded_in(evidence: str, text: str) -> bool:
    """Public accessor: returns True if evidence matches exact or fuzzy criteria."""
    return _is_grounded(evidence, _norm(text)) in ("exact", "fuzzy")


def value_grounded_in(value: float, text: str) -> bool:
    """Verifies that the numeric literal explicitly appears in the source text."""
    for probe in (f"{value:.4f}".rstrip("0").rstrip("."), f"{value:.3f}",
                  f"{value:.2f}", f"{value:.1f}"):
        if re.search(r"(?<![0-9.])" + re.escape(probe) + r"(?![0-9])", text):
            return True
    return False


SUPERSCRIPT = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789")


def _numbers_in(text: str) -> list[float]:
    """Extracts floating point values from source text, normalizing typographic exponents."""
    text = re.sub(r"[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+",
                  lambda m: m.group().translate(SUPERSCRIPT), text.replace("\u207b", "-"))
    out: list[float] = []
    for raw in re.findall(r"[-+]?\d*\.?\d+(?:\s*[x×]\s*10\s*[-−]?\s*\d+|e[-+]?\d+)?", text, re.I):
        cleaned = re.sub(r"\s+", "", raw).replace("−", "-").replace("×", "x")
        try:
            if "x10" in cleaned.lower():
                mantissa, _, exponent = cleaned.lower().partition("x10")
                out.append(float(mantissa) * 10 ** int(exponent))
            else:
                out.append(float(cleaned))
        except ValueError:
            continue
    for mantissa, exponent in re.findall(r"\b10\s*[-−]\s*(\d+)()", text):
        try:
            out.append(10.0 ** -int(mantissa))
        except ValueError:
            continue
    return out


# =============================================================================
#                    ********* METRIC EVALUATION *********                   
#     Baseline scoring algorithms producing precision, recall, and F1.       
# =============================================================================

def _supports(value: float, sentence: str) -> tuple[bool, str]:
    """Checks if a value is derivable from the sentence numbers via unit conversion."""
    candidates = _numbers_in(sentence)
    if not candidates:
        return False, ""

    def close(a: float, b: float) -> bool:
        return abs(b) > 0 and abs(a - b) / abs(b) <= REL_TOL

    for number in candidates:
        if close(number, value):
            return True, "direct"
        for factor, name in ((RY_TO_EV, "Ry->eV"),
                             (RY_BOHR_TO_EV_ANG, "Ry/bohr->eV/A"),
                             (NM_TO_ANG, "nm->A"),
                             (HARTREE_TO_EV, "Ha->eV")):
            if close(number * factor, value):
                return True, name
            if close(number / factor, value):
                return True, f"{name} (inverse)"
    return False, ""


def evaluate_one(rec: Extraction, text: str) -> list[FieldCheck]:
    """Verifies all reported fields of a single record against the paper text."""
    haystack = _norm(text)
    checks: list[FieldCheck] = []
    for name in type(rec.method).model_fields:
        entry = getattr(rec.method, name)
        if not entry.reported:
            continue
        evidence = (entry.evidence or "").strip()
        match = _is_grounded(evidence, haystack)
        check = FieldCheck(
            paper_key=rec.paper_key, field=name, value=entry.value,
            grounded=match in ("exact", "fuzzy"), match=match,
            evidence=evidence[:160],
        )
        if (check.grounded and isinstance(entry.value, (int, float))
                and not isinstance(entry.value, bool)):
            check.supported, check.via = _supports(float(entry.value), evidence)
        checks.append(check)
    return checks


def run(out_dir: Optional[Path] = None) -> dict:
    """Evaluates all extractions against their cached full text."""
    out_dir = Path(out_dir or PROCESSED)
    path = INTERIM / "extracted.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `acv extract` first")

    report = EvalReport()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = Extraction(**json.loads(line))
        report.n_records += 1
        text_file = _text_path(rec.paper_key)
        if not text_file.exists():
            continue
        report.n_with_text += 1
        report.checks.extend(
            evaluate_one(rec, text_file.read_text(encoding="utf-8", errors="replace"))
        )

    stats = report.summary()
    from ..settings import config_fingerprint

    stats["provenance"] = {"config_fingerprint": config_fingerprint()}
    (out_dir / "evaluation.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with open(out_dir / "evaluation_checks.jsonl", "w", encoding="utf-8") as fh:
        for c in report.checks:
            fh.write(json.dumps(c.__dict__, default=str) + "\n")
    log.info("evaluation: %s", stats["grounding"])
    return stats
