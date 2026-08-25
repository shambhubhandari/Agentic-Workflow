"""Tier 2: loader module.

Provides strict, deterministic logic and strict typing for loader operations.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

# =============================================================================
#                    ********* CONSTANTS & THRESHOLDS *********                  
#         Filesystem paths and configuration exceptions for prompt loading.    
# =============================================================================

CREWS_DIR = Path(__file__).resolve().parent


class CrewConfigError(RuntimeError):
    """A crew's config is missing or malformed."""


# =============================================================================
#                      ********* YAML PARSING *********                    
#          Strict, deterministic loading of YAML crew configurations.            
# =============================================================================

def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CrewConfigError(f"missing crew config: {path}")
    try:
        import yaml
    except ImportError as exc:                       # pragma: no cover
        raise CrewConfigError("pyyaml is required to load crew configs") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CrewConfigError(f"crew config is not a mapping: {path}")
    return data


@lru_cache(maxsize=None)
def load(crew: str) -> dict[str, Any]:
    """Return {'agents': ..., 'tasks': ..., 'version': ..., 'digest': ...} for a crew."""
    base = CREWS_DIR / crew / "config"
    agents = _read_yaml(base / "agents.yaml")
    tasks = _read_yaml(base / "tasks.yaml")

    blob = (base / "agents.yaml").read_bytes() + (base / "tasks.yaml").read_bytes()
    return {
        "agents": {k: v for k, v in agents.items() if k != "version"},
        "tasks": {k: v for k, v in tasks.items() if k != "version"},
        "version": f"{agents.get('version', 0)}.{tasks.get('version', 0)}",
        "digest": hashlib.sha256(blob).hexdigest()[:16],
    }


# =============================================================================
#                      ********* PROMPT ASSEMBLY *********                    
#         Concatenation of role and instructions with strict versioning.       
# =============================================================================

def prompt(crew: str, agent_key: str, task_key: str) -> tuple[str, str]:
    """Assemble the system prompt for one agent+task, returning it with its version."""
    cfg = load(crew)
    try:
        agent = cfg["agents"][agent_key]
        task = cfg["tasks"][task_key]
    except KeyError as exc:
        raise CrewConfigError(f"{crew}: no such key {exc}") from exc

    text = (
        f"You are: {agent.get('role', '').strip()}\n\n"
        f"Your goal: {agent.get('goal', '').strip()}\n\n"
        f"{agent.get('backstory', '').strip()}\n\n"
        f"---\n\n"
        f"{task.get('description', '').strip()}\n\n"
        f"Expected output: {task.get('expected_output', '').strip()}\n"
    )
    return text, f"{crew}/{cfg['version']}/{cfg['digest']}"
