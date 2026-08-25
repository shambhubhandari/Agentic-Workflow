"""System: audit_log module.

Provides strict, deterministic logic and strict typing for audit_log operations.
"""
from __future__ import annotations

# =============================================================================
#                      ********* LIFECYCLE HOOKS *********                     
#                       Strict definitions for audit_log.                      
# =============================================================================

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..settings import PROCESSED

log = logging.getLogger(__name__)

LOG_PATH = PROCESSED / "agent_log.jsonl"
_LOCK = threading.Lock()

# Log prompt metadata via version hash to prevent bloat.
MAX_PROMPT_TAIL = 1200
MAX_RESPONSE = 4000


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"


def record(
    agent: str,
    prompt_id: str,
    *,
    model: str,
    prompt_tail: Optional[str] = None,
    response: Optional[Any] = None,
    proposed: Optional[dict] = None,
    applied: Optional[dict] = None,
    overrides: Optional[list[str]] = None,
    seconds: Optional[float] = None,
    context: Optional[dict] = None,
    error: Optional[str] = None,
) -> dict:
    """Append one agent exchange to the audit log.

    `proposed` is what the agent returned; `applied` is what survived the code gates;
    `overrides` names each gate that fired. When those differ, the difference is the
    system working.
    """
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": agent,
        "prompt_id": prompt_id,
        "model": model,
        "seconds": round(seconds, 2) if seconds is not None else None,
        "context": context or {},
        "prompt_tail": _truncate(prompt_tail, MAX_PROMPT_TAIL),
        "response": _truncate(
            response if isinstance(response, str) else json.dumps(response, default=str),
            MAX_RESPONSE,
        ),
        "proposed": proposed,
        "applied": applied,
        "overrides": overrides or [],
        "error": error,
    }

    # A one-line human summary, so `tail -f` is readable without jq.
    bits = [f"{agent}"]
    if context:
        bits.append(" ".join(f"{k}={v}" for k, v in list(context.items())[:2]))
    if overrides:
        bits.append(f"OVERRIDDEN({len(overrides)}): {overrides[0][:60]}")
    if error:
        bits.append(f"ERROR {error[:60]}")
    entry["summary"] = " | ".join(bits)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, default=str)
    with _LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    # Mirror to the normal logger so a console run shows agent activity live.
    if overrides:
        log.warning("[%s] %s", agent, entry["summary"])
    else:
        log.info("[%s] %s", agent, entry["summary"])
    return entry


def read_all(path: Optional[Path] = None) -> list[dict]:
    path = Path(path or LOG_PATH)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarise(path: Optional[Path] = None) -> dict:
    """How often did each agent run, and how often was it overridden?"""
    from collections import Counter

    entries = read_all(path)
    by_agent: dict[str, dict] = {}
    for e in entries:
        a = by_agent.setdefault(e["agent"], {"calls": 0, "overridden": 0, "seconds": 0.0})
        a["calls"] += 1
        a["overridden"] += 1 if e.get("overrides") else 0
        a["seconds"] += e.get("seconds") or 0.0
    return {
        "n_calls": len(entries),
        "by_agent": by_agent,
        "override_reasons": dict(
            Counter(o for e in entries for o in (e.get("overrides") or []))
        ),
    }
