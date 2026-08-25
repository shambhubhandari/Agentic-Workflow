"""Tier 0: SIESTA directive knowledge extraction.

Parses SIESTA source code to extract directive existence, defaults, and units.
Provides deterministic parameter validation and default-inference to 
ensure exact reproductive constraints.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# =============================================================================
#                    ********* CONSTANTS & CACHING *********                  
#         Filesystem paths and memoized index retrieval routines.              
# =============================================================================

INDEX_PATH: Path = Path(__file__).resolve().parent / "siesta_directives.json"


@lru_cache(maxsize=1)
def _index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        log.warning("no SIESTA directive index at %s; run build_siesta_index", INDEX_PATH)
        return {"siesta_version": None, "directives": {}, "_lower": {}}
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    data["_lower"] = {k.lower(): k for k in data.get("directives", {})}
    return data


def version() -> Optional[str]:
    return _index().get("siesta_version")

# =============================================================================
#                   ********* FDF QUERY INTERFACE *********                  
#      Case-insensitive lookups, search bounds, and default aggregation.       
# =============================================================================

def exists(name: str) -> bool:
    return (name or "").strip().lower() in _index()["_lower"]


def lookup(name: str) -> dict[str, Any]:
    """Full record for a directive, or a not-found result naming close matches."""
    idx = _index()
    key = idx["_lower"].get((name or "").strip().lower())
    if key is None:
        return {
            "name": name,
            "exists": False,
            "siesta_version": idx.get("siesta_version"),
            "note": (
                f"{name!r} is not read by SIESTA {idx.get('siesta_version')}. Setting it "
                "would be silently ignored."
            ),
            "did_you_mean": search(name, limit=5).get("matches", []),
        }
    entry = dict(idx["directives"][key])
    entry["exists"] = True
    entry["siesta_version"] = idx.get("siesta_version")
    return entry


def search(fragment: str, limit: int = 20) -> dict[str, Any]:
    """Find directives whose name contains a fragment."""
    frag = (fragment or "").strip().lower().replace(" ", "")
    if not frag:
        return {"fragment": fragment, "matches": []}
    hits = [k for k in _index()["directives"] if frag in k.lower().replace(" ", "")]
    hits.sort(key=lambda k: (len(k), k.lower()))
    return {"fragment": fragment, "n_hits": len(hits), "matches": hits[:limit]}


def defaults_for(names: list[str]) -> dict[str, Any]:
    """Defaults for several directives at once, for reporting what was left unset."""
    return {
        n: {
            "default": rec.get("default"),
            "unit": rec.get("unit"),
            "type": rec.get("type"),
        }
        for n in names
        if (rec := lookup(n)).get("exists")
    }

# =============================================================================
#                     ********* VALIDATION LOGIC *********                   
#        Strict filtering of proposed parameters against the SIESTA binary.    
# =============================================================================

def validate(directives: dict[str, Any]) -> dict[str, Any]:
    """Split a proposed directive set into what SIESTA will read and what it will ignore."""
    known: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    for name, value in (directives or {}).items():
        (known if exists(name) else unknown)[name] = value
    return {
        "known": known,
        "unknown": unknown,
        "n_known": len(known),
        "n_unknown": len(unknown),
        "note": (
            "all directives recognised" if not unknown else
            f"{len(unknown)} directive(s) not read by SIESTA {version()}: "
            + ", ".join(sorted(unknown))
        ),
    }
