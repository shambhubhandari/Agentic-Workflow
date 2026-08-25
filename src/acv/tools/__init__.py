"""Agent-callable tools.

Importing this package registers every tool. Agents reach them only through
`registry.call`, which enforces the per-agent allow-list and logs the invocation.
"""

# =============================================================================
#                        ********* AGENT TOOLS *********                       
#                       Strict definitions for __init__.                       
# =============================================================================

from . import inspect as _inspect          # noqa: F401  (registers the tools)
from .registry import ToolRefused, call, declarations, for_agent, get, register

__all__ = ["call", "declarations", "for_agent", "get", "register", "ToolRefused"]
