"""Pipeline consensus mechanics for multi-pass extraction merging.

Implements union-based record merging to combat LLM attention-cache 
non-determinism. Bypasses majority-voting to prevent whole-paper collapse 
from manufacturing false absences.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ..settings import INTERIM
from ..types import Extraction
from .evaluate import grounded_in
from .fetch import _text_path

log = logging.getLogger(__name__)

# =============================================================================
#                     ********* MERGE ALGORITHMS *********                   
#       Union-based aggregation strategies for multi-pass extractions.       
# =============================================================================

PLANE_WAVE_CODES = {"vasp", "quantum_espresso", "castep", "abinit", "wien2k", "gaussian"}
NAO_CODES = {"siesta", "openmx", "dmol3"}


def _reported_code(rec: dict) -> str:
    """Extracts the software code name from a record if reported."""
    entry = rec["method"].get("code") or {}
    value = entry.get("value")
    return str(value).lower() if entry.get("reported") and value else ""


def regate(rec: dict, text: str) -> tuple[dict, int]:
    """Drops any reported field whose evidence is absent from the source text."""
    cut = 0
    for entry in rec["method"].values():
        if isinstance(entry, dict) and entry.get("reported") and entry.get("evidence"):
            if not grounded_in(entry["evidence"], text):
                entry["reported"] = False
                entry["value"] = None
                cut += 1
    return rec, cut


def resolve_collisions(rec: dict) -> list[str]:
    """Resolves mutually exclusive basis-set cutoffs by checking the reported code."""
    fixed: list[str] = []
    method = rec["method"]
    pw, mesh = method.get("plane_wave_cutoff_ev"), method.get("mesh_cutoff_ry")
    
    if not (pw and mesh and pw.get("reported") and mesh.get("reported")):
        return fixed
        
    code = _reported_code(rec)
    drop = ("mesh_cutoff_ry" if code in PLANE_WAVE_CODES
            else "plane_wave_cutoff_ev" if code in NAO_CODES else None)
            
    if drop is None:
        return fixed
        
    method[drop]["reported"] = False
    method[drop]["value"] = None
    fixed.append(f"{drop} (code={code})")
    return fixed


def union(records: list[dict]) -> dict:
    """Merges extractions, preferring the first reported instance of any field."""
    out = json.loads(json.dumps(records[0]))
    for other in records[1:]:
        for name, entry in other["method"].items():
            if (isinstance(entry, dict) and entry.get("reported")
                    and not out["method"][name].get("reported")):
                out["method"][name] = entry
        for flag in ("is_computational", "is_pentagonal_2d"):
            if other.get(flag) and not out.get(flag):
                out[flag] = True
        if not (out.get("claims") or []) and (other.get("claims") or []):
            out["claims"] = other["claims"]
    return out


# =============================================================================
#                       ********* PUBLIC API *********                       
#           Entrypoints for cross-pass consolidation and statistics.         
# =============================================================================

def stability(passes: list[dict[str, dict]]) -> dict[str, int | float | None]:
    """Computes field agreement stability (intersection over union) across passes."""
    keys = set(passes[0])
    for p in passes[1:]:
        keys &= set(p)
        
    def fields(rec: dict) -> set[str]:
        return {n for n, e in rec["method"].items() if isinstance(e, dict) and e.get("reported")}
        
    in_all = in_any = 0
    for key in keys:
        sets = [fields(p[key]) for p in passes]
        in_all += len(set.intersection(*sets))
        in_any += len(set.union(*sets))
        
    return {
        "n_passes": len(passes), "n_papers": len(keys),
        "fields_in_all_passes": in_all, "fields_in_any_pass": in_any,
        "stability": (in_all / in_any) if in_any else None
    }


def combine(pass_paths: list[Path], out_path: Optional[Path] = None) -> dict[str, int | float | None]:
    """Re-gates passes against source text, unions them, and resolves collisions."""
    out_path = Path(out_path or (INTERIM / "extracted.jsonl"))
    loaded: list[dict[str, dict]] = []
    total_cut = 0
    
    for path in pass_paths:
        bykey: dict[str, dict] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            text_file = _text_path(rec["paper_key"])
            if text_file.exists():
                rec, cut = regate(rec, text_file.read_text(encoding="utf-8", errors="replace"))
                total_cut += cut
            bykey[rec["paper_key"]] = rec
        loaded.append(bykey)

    stats = stability(loaded) if len(loaded) > 1 else {"n_passes": 1}
    keys = set(loaded[0])
    for p in loaded[1:]:
        keys &= set(p)

    merged, n_collisions, n_unresolved = [], 0, 0
    for key in sorted(keys):
        rec = union([p[key] for p in loaded])
        fixes = resolve_collisions(rec)
        n_collisions += len(fixes)
        
        method = rec["method"]
        if (method.get("plane_wave_cutoff_ev", {}).get("reported")
                and method.get("mesh_cutoff_ry", {}).get("reported")):
            n_unresolved += 1
        merged.append(rec)

    out_path.write_text("\n".join(json.dumps(r) for r in merged) + "\n", encoding="utf-8")
    
    n_fields = sum(1 for r in merged for e in r["method"].values()
                   if isinstance(e, dict) and e.get("reported"))
                   
    stats.update({
        "ungrounded_fields_cut": total_cut, "reported_fields": n_fields,
        "collisions_resolved": n_collisions,
        "collisions_unresolved": n_unresolved
    })
    log.info("consensus: %s", stats)
    return stats


def load_passes(pass_dir: Optional[Path] = None) -> list[Path]:
    """Lists completed extraction passes on disk sequentially."""
    pass_dir = Path(pass_dir or (INTERIM / "passes"))
    return sorted(pass_dir.glob("extracted.pass*.jsonl"))


def as_records(path: Optional[Path] = None) -> list[Extraction]:
    """Deserializes an extraction JSONL file into typed Extraction instances."""
    path = Path(path or (INTERIM / "extracted.jsonl"))
    return [Extraction(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
