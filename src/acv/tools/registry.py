"""System: registry module.

Provides strict, deterministic logic and strict typing for registry operations.
"""
from __future__ import annotations

# =============================================================================
#                        ********* AGENT TOOLS *********                       
#                       Strict definitions for registry.                       
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    schema: dict
    mutates: bool = False          # does calling this change state or spend compute?


_REGISTRY: dict[str, Tool] = {}

# Which agent may call what. Deliberately restrictive: an agent gets the minimum set
# that lets it do its job.
ALLOWED: dict[str, set[str]] = {
    "translator": {
        "list_pseudopotentials", "check_basis_supported", "literature_methods",
        "siesta_directive", "siesta_search",
    },
    "convergence": {
        "convergence_status", "estimate_cost", "literature_methods",
        "siesta_directive",
    },
    "diagnostician": {
        "search_output", "machine_resources", "siesta_directive",
    },
    "critic": {
        # The Critic may inspect everything and change nothing.
        "read_structure", "check_symmetry", "convert_units",
        "calibration_for_structure", "convergence_status", "search_output",
        "expected_symmetry", "literature_claims", "literature_methods",
    },
}


def register(
    name: str, description: str, schema: dict, mutates: bool = False
) -> Callable:
    """Decorator registering a function as an agent-callable tool."""

    def wrap(func: Callable) -> Callable:
        _REGISTRY[name] = Tool(name, description, func, schema, mutates)
        return func

    return wrap


def get(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def for_agent(agent: str) -> list[Tool]:
    """Tools this agent is permitted to call."""
    names = ALLOWED.get(agent, set())
    return [t for n, t in sorted(_REGISTRY.items()) if n in names]


def declarations(agent: str) -> list[dict]:
    """Function declarations for the tools this agent may call."""
    return [
        {"name": t.name, "description": t.description, "parameters": t.schema}
        for t in for_agent(agent)
    ]


class ToolRefused(RuntimeError):
    """An agent called a tool it is not permitted to use."""


def call(agent: str, name: str, arguments: dict) -> dict:
    """Invoke a tool on behalf of an agent, enforcing the allow-list.

    Returns a dict result. Errors are returned rather than raised so the agent can see
    what went wrong and adjust -- an exception would simply end its turn.
    """
    import time

    from ..hooks import audit_log

    if name not in ALLOWED.get(agent, set()):
        audit_log.record(
            f"{agent}:tool", "registry", model="-",
            context={"tool": name}, overrides=[f"refused: {agent} may not call {name}"],
            error="not permitted",
        )
        raise ToolRefused(f"agent {agent!r} may not call tool {name!r}")

    tool = get(name)
    if tool is None:
        return {"error": f"no such tool: {name}"}

    started = time.time()
    try:
        result = tool.func(**(arguments or {}))
        error = None
    except Exception as exc:                          # noqa: BLE001
        result = {"error": f"{type(exc).__name__}: {exc}"}
        error = str(exc)

    audit_log.record(
        f"{agent}:tool", "registry", model="-",
        response=result, context={"tool": name, "args": arguments},
        seconds=time.time() - started, error=error,
    )
    return result if isinstance(result, dict) else {"result": result}
