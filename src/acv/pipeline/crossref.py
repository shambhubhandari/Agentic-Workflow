"""Tier 0.5: Cross-reference extracted claims against public DFT databases.

Queries JARVIS-DFT and 2DMatpedia for pentagonal polymorphs matching 
extracted compositions, evaluating claimed lattice and energy parameters 
against known reference structures.
"""

from __future__ import annotations

import json
import logging
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from ..settings import INTERIM, PROCESSED
from ..types import Extraction, ExtractionStatus
from .normalize import PropertyKind, normalize_formula, property_kind, reduced_formula

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & SETTINGS *********                  
#        Structural constraints, property mappings, and API configuration.     
# =============================================================================

PROVIDERS: dict[str, str] = {
    "2dmatpedia": "http://optimade.2dmatpedia.org/v1/structures",
    "jarvis_dft": "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures",
}

REQUEST_DELAY_S: float = 0.5
MAX_STRUCTURES: int = 50

MIN_VACUUM_ANG: float = 8.0
LATTICE_TOL_FRAC: float = 0.02
LATTICE_KINDS: set[PropertyKind] = {PropertyKind.LATTICE_A, PropertyKind.LATTICE_B, PropertyKind.LATTICE_UNSPEC}

JARVIS_SENTINEL: float = -99999

PROPERTY_FIELDS: dict[PropertyKind, tuple[str, float, str]] = {
    PropertyKind.FORMATION_ENERGY: ("_jarvis_formation_energy_peratom", 0.15, "eV/atom"),
    PropertyKind.BULK_MODULUS:     ("_jarvis_bulk_modulus_kv",          20.0, "GPa"),
}

ELEMENTAL_FORMATION_MAX_EV: float = 3.0


@dataclass
class Candidate:
    provider: str
    entry_id: str
    formula: str
    a: float
    b: float
    c: float
    nsites: int
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimCheck:
    paper_key: str
    doi: Optional[str]
    property: str
    kind: str
    formula: Optional[str]
    reduced: Optional[str]
    claimed: Optional[float]
    unit: Optional[str]
    n_candidates: int = 0
    best_match: Optional[str] = None
    best_value: Optional[float] = None
    rel_error: Optional[float] = None
    verdict: str = "no_reference"
    note: Optional[str] = None


@dataclass
class CrossRefReport:
    checks: list[ClaimCheck] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        from collections import Counter
        verdicts = Counter(c.verdict for c in self.checks)
        matched = [c for c in self.checks if c.rel_error is not None]
        return {
            "n_claims_checked": len(self.checks),
            "verdicts": dict(verdicts),
            "n_with_reference": len(matched),
            "median_rel_error_pct": (
                round(100 * sorted(c.rel_error for c in matched)[len(matched) // 2], 2)
                if matched else None
            ),
        }

# =============================================================================
#                    ********* NETWORK PROCUREMENT *********                 
#       OPTIMADE queries and pentagonal geometry structural validators.      
# =============================================================================

def _cache_path() -> Path:
    return INTERIM / "optimade_cache_v2.json"


def _load_cache() -> dict[str, list[dict[str, Any]]]:
    path = _cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_cache(cache: dict[str, list[dict[str, Any]]]) -> None:
    _cache_path().parent.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache, indent=2))


def _lattice_lengths(vectors: Optional[list[list[float]]]) -> Optional[list[float]]:
    if not vectors:
        return None
    try:
        return [sum(v * v for v in row) ** 0.5 for row in vectors]
    except TypeError:
        return None


def _is_penta_prototype(attrs: dict[str, Any]) -> bool:
    from .calibrate import is_penta_prototype
    return is_penta_prototype(attrs)


def query_structures(reduced: str, cache: dict[str, list[dict[str, Any]]]) -> list[Candidate]:
    """Retrieve 2D pentagonal reference structures from OPTIMADE endpoints."""
    if reduced in cache:
        return [Candidate(**c) for c in cache[reduced]]

    found: list[Candidate] = []
    for name, base_url in PROVIDERS.items():
        try:
            response = requests.get(
                base_url,
                params={
                    "filter": f'chemical_formula_reduced="{reduced}"',
                    "page_limit": MAX_STRUCTURES,
                    "response_fields": "cartesian_site_positions,species_at_sites,lattice_vectors,nsites,"
                                       "chemical_formula_reduced,_jarvis_formation_energy_peratom,"
                                       "_jarvis_bulk_modulus_kv",
                },
                timeout=45,
            )
            time.sleep(REQUEST_DELAY_S)
            if response.status_code != 200:
                log.warning("%s returned %s for %s", name, response.status_code, reduced)
                continue
            for entry in response.json().get("data", []):
                attrs = entry.get("attributes", {})
                lengths = _lattice_lengths(attrs.get("lattice_vectors"))
                if not lengths:
                    continue
                if max(lengths) < MIN_VACUUM_ANG:
                    continue
                if not _is_penta_prototype(attrs):
                    continue
                a, b, c = sorted(lengths)
                props = {
                    k: v for k, v in attrs.items()
                    if k.startswith("_jarvis_")
                    and v not in (JARVIS_SENTINEL, None, "", [])
                }
                found.append(
                    Candidate(
                        provider=name,
                        entry_id=str(entry.get("id")),
                        formula=attrs.get("chemical_formula_reduced", reduced),
                        a=a, b=b, c=c,
                        nsites=attrs.get("nsites") or 0,
                        properties=props,
                    )
                )
        except requests.RequestException as exc:
            log.warning("%s query failed for %s: %s", name, reduced, exc)

    cache[reduced] = [vars(c) for c in found]
    _save_cache(cache)
    return found


# =============================================================================
#                   ********* CLAIM VERIFICATION *********                   
#      Comparison logic mapping paper claims to DFT reference structures.    
# =============================================================================

def check_claim(rec: Extraction, claim: Any, cache: dict[str, list[dict[str, Any]]]) -> ClaimCheck:
    kind = property_kind(claim.property)
    formula = normalize_formula(claim.material_formula)
    reduced = reduced_formula(formula) if formula else None

    check = ClaimCheck(
        paper_key=rec.paper_key, doi=rec.doi, property=claim.property,
        kind=kind.value, formula=formula, reduced=reduced,
        claimed=claim.value, unit=claim.unit,
    )

    if claim.value is None or not reduced:
        check.verdict = "skipped"
        return check
    if kind not in LATTICE_KINDS and kind not in PROPERTY_FIELDS:
        check.verdict = "skipped"
        return check

    candidates = query_structures(reduced, cache)
    check.n_candidates = len(candidates)
    if not candidates:
        check.verdict = "no_reference"
        return check

    if kind in PROPERTY_FIELDS:
        field_name, tolerance, _unit = PROPERTY_FIELDS[kind]

        if (
            kind is PropertyKind.FORMATION_ENERGY
            and reduced
            and len(re.findall(r"[A-Z][a-z]?", reduced)) == 1
            and abs(claim.value) > ELEMENTAL_FORMATION_MAX_EV
        ):
            check.verdict = "definition_mismatch"
            check.note = (
                f"labelled formation energy but |{claim.value}| exceeds "
                f"{ELEMENTAL_FORMATION_MAX_EV} eV for an elemental system; "
                "likely a cohesive or atomisation energy"
            )
            return check

        best_prop = None
        for cand in candidates:
            value = cand.properties.get(field_name)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            diff = abs(claim.value - value)
            if best_prop is None or diff < best_prop[0]:
                best_prop = (diff, cand, value)

        if best_prop is None:
            check.verdict = "no_reference"
            return check

        abs_diff, cand, matched = best_prop
        check.best_match = f"{cand.provider}:{cand.entry_id}"
        check.best_value = matched
        check.rel_error = abs_diff / abs(matched) if matched else None
        check.verdict = "corroborated" if abs_diff <= tolerance else "unmatched"
        return check

    best = None
    for cand in candidates:
        for value in (cand.a, cand.b):
            err = abs(claim.value - value) / value if value else None
            if err is not None and (best is None or err < best[0]):
                best = (err, cand, value)

    if best is None:
        check.verdict = "no_reference"
        return check

    check.rel_error, cand, matched_value = best
    check.best_match = f"{cand.provider}:{cand.entry_id}"
    check.best_value = matched_value
    check.verdict = "corroborated" if check.rel_error <= LATTICE_TOL_FRAC else "unmatched"
    return check


def run(records: Optional[list[Extraction]] = None, out_dir: Optional[Path] = None) -> dict[str, Any]:
    from . import extract as extract_mod

    records = records if records is not None else extract_mod.load()
    records = [r for r in records if r.status == ExtractionStatus.OK]
    out_dir = out_dir or PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = _load_cache()
    report = CrossRefReport()
    for rec in records:
        for claim in rec.claims:
            report.checks.append(check_claim(rec, claim, cache))

    with open(out_dir / "crossref.jsonl", "w", encoding="utf-8") as fh:
        for check in report.checks:
            fh.write(json.dumps(vars(check)) + "\n")

    stats = report.summary()
    (out_dir / "crossref_summary.json").write_text(json.dumps(stats, indent=2))
    log.info("cross-reference: %s", stats)
    return stats
