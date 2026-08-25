"""System: registry module.

Provides strict, deterministic logic and strict typing for registry operations.
"""
from __future__ import annotations

# =============================================================================
#                   ********* VERIFICATION METRICS *********                   
#                       Strict definitions for registry.                       
# =============================================================================

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .. import settings as S

PASS, FAIL, NEEDS_TEXT, SKIP = "PASS", "FAIL", "NEEDS-FULLTEXT", "SKIP"


@dataclass(frozen=True)
class Entry:
    id: str
    value: Any
    section: str
    disposition: str          # promote | pinned | external
    tolerance: float = 0.0
    unit: str = ""
    macro: str = ""           # generated macro this value corresponds to, if any
    population: str = ""      # which set it is computed over; see the registry header
    verify: str = ""
    artefacts: tuple[str, ...] = ()
    evidence: str = ""
    note: str = ""


@dataclass(frozen=True)
class Result:
    entry: Entry
    status: str
    observed: Any = None
    detail: str = ""


def load(path: Path | None = None) -> list[Entry]:
    path = path or S.REPORTED_VALUES
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Entry(**{**e, "artefacts": tuple(e.get("artefacts", []))}) for e in raw]


def _call(target: str) -> Any:
    """Resolve 'package.module:function' and call it with no arguments."""
    module_name, _, function_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)()


def check(entry: Entry) -> Result:
    if entry.disposition == "external":
        return Result(entry, SKIP, detail="third-party value, cited not recomputed")
    if entry.disposition == "pinned":
        evidence = S.PROJECT_ROOT / entry.evidence.split(":")[0] if entry.evidence else None
        if evidence and evidence.exists():
            return Result(entry, SKIP, detail=f"pinned to {entry.evidence}")
        return Result(entry, FAIL, detail="pinned value lacks corresponding evidence file")
    if not entry.verify:
        return Result(entry, FAIL, detail="promote entry with no verify function")

    missing = [a for a in entry.artefacts if not (S.PROJECT_ROOT / a).exists()]
    if missing:
        needs_text = any("fulltext" in m for m in missing)
        return Result(entry, NEEDS_TEXT if needs_text else FAIL,
                      detail=f"missing {missing[0]}")
    try:
        observed = _call(entry.verify)
    except FileNotFoundError as exc:
        name = str(exc)
        return Result(entry, NEEDS_TEXT if "fulltext" in name else FAIL, detail=name[:70])
    except Exception as exc:                                   # noqa: BLE001
        return Result(entry, FAIL, detail=f"{type(exc).__name__}: {exc}"[:70])

    if observed is None:
        return Result(entry, NEEDS_TEXT, detail="recompute returned no value")
    if isinstance(entry.value, (int, float)) and isinstance(observed, (int, float)):
        ok = abs(float(observed) - float(entry.value)) <= entry.tolerance
    else:
        ok = str(observed) == str(entry.value)
    return Result(entry, PASS if ok else FAIL, observed=observed)


def run(entries: list[Entry] | None = None) -> list[Result]:
    return [check(e) for e in (entries if entries is not None else load())]
