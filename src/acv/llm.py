"""Schema-constrained LLM provider interface.

Executes local inference strictly constrained to a provided JSON schema (via Pydantic). 
Execution runs locally via Ollama to guarantee absolute reproducibility and zero reliance 
on transient upstream model APIs. Constrained decoding enforces the structural shape of 
the response, while semantic correctness is delegated to downstream validation blocks.

Functions:
    agent_model: Resolves the appropriate provider and model for a given agent.
    structured: Executes a prompt and returns a validated Pydantic model.
    health: Reports provider readiness.
"""

# =============================================================================
#                  ********* IMPORTS & CONSTANTS *********                   
#         Core dependencies and static physical context boundaries.          
# =============================================================================

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TypeVar

from pydantic import BaseModel
import requests

from .settings import (AGENT_MODEL, AGENT_MODELS, AGENT_PROVIDER, AGENT_PROVIDERS,
                       LLM_CHARS_PER_TOKEN, LLM_CTX_BUCKET, LLM_CTX_HEADROOM, LLM_CTX_MAX,
                       LLM_CTX_MIN,
                       LLM_KEEP_ALIVE, LLM_NUM_PREDICT, LLM_TEMPERATURE, LLM_THINK,
                       LLM_TIMEOUT_S, OLLAMA_HOST, OLLAMA_MODEL)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Context retention configuration for cyclic SIESTA invocations.
# Limits VRAM retention to 5 minutes to prevent out-of-memory contention 
# between resident models and physical DFT relaxations.
OLLAMA_KEEP_ALIVE = LLM_KEEP_ALIVE
OLLAMA_NUM_PREDICT = LLM_NUM_PREDICT
OLLAMA_TIMEOUT_S = LLM_TIMEOUT_S
CHARS_PER_TOKEN = LLM_CHARS_PER_TOKEN
OLLAMA_CTX_MAX = LLM_CTX_MAX


# =============================================================================
#                       ********* EXCEPTIONS *********                       
#      Custom error models for network boundaries and prompt truncation.     
# =============================================================================

class ProviderError(Exception):
    """Raised when the LLM provider fails to fulfill a request."""
    pass


class ContextTooLarge(Exception):
    """Raised when the prompt strictly exceeds maximum token context bounds."""
    pass


# =============================================================================
#              ********* CONTEXT & SCHEMA COMPILATION *********              
#    Algorithms for grammar subsetting and safe token context allocation.    
# =============================================================================

def _context_tokens(text: str) -> int:
    """Calculates minimal safe context window allocated in distinct buckets."""
    chars = len(text)
    exact = int(chars / CHARS_PER_TOKEN) + LLM_CTX_HEADROOM
    return max(LLM_CTX_MIN, min(OLLAMA_CTX_MAX, 
               ((exact + LLM_CTX_BUCKET - 1) // LLM_CTX_BUCKET) * LLM_CTX_BUCKET))


def _was_truncated(body: dict[str, Any], limit: int) -> bool:
    """Verifies if the response indicates context exhaustion."""
    if body.get("done_reason") == "length":
        return True
    seen = body.get("prompt_eval_count")
    if seen and limit and abs(seen - limit // 2) <= 16:
        return True
    return False


def _drop_unsupported(schema: dict[str, Any]) -> dict[str, Any]:
    """Prunes unsupported JSON schema definitions prior to API injection."""
    if "title" in schema:
        del schema["title"]
    if "description" in schema:
        del schema["description"]
    for val in schema.values():
        if isinstance(val, dict):
            _drop_unsupported(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    _drop_unsupported(item)
    return schema


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inlines schema references to satisfy Ollama strict decoding limits."""
    if "$defs" not in schema:
        return schema
    defs = schema.pop("$defs")

    def _replace(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"].split("/")[-1]
                return _replace(defs[ref])
            return {k: _replace(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_replace(x) for x in node]
        return node

    return _replace(schema)


def require_fields(schema: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    """Enforces specific required fields inside the provided schema constraint."""
    if "properties" in schema:
        schema["required"] = sorted(
            list(set(schema.get("required", [])).union(fields.intersection(schema["properties"])))
        )
    return schema


def _require_all(schema: dict[str, Any]) -> dict[str, Any]:
    """Enforces exhaustive field requirements, blocking model omissions."""
    if schema.get("type") == "object" and "properties" in schema:
        schema["required"] = sorted(list(schema["properties"].keys()))
        for prop in schema["properties"].values():
            _require_all(prop)
    elif schema.get("type") == "array" and "items" in schema:
        _require_all(schema["items"])
    return schema


def _schema_instructions(schema: dict[str, Any]) -> str:
    """Injects schema documentation into the textual prompt."""
    if "properties" not in schema:
        return ""
    pairs = [(k, v.get("description", "")) for k, v in schema["properties"].items() 
             if v.get("description")]
    if not pairs:
        return ""
    lines = ["", "FIELD REQUIREMENTS (each field of your JSON answer must satisfy these):"]
    lines += [f"  - {name}: {desc}" for name, desc in pairs]
    return "\n".join(lines) + "\n"


# =============================================================================
#                   ********* PROVIDER EXECUTION *********                   
#     Synchronous network bindings and retry logic for Ollama endpoints.     
# =============================================================================

def _execute_request(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Dispatches a synchronous request to the underlying API.

    Args:
        payload: Compiled request configuration.
        model: Target model identifier for localized error reporting.

    Returns:
        dict[str, Any]: Raw JSON response from the endpoint.
        
    Raises:
        ProviderError: Extracted for network failures, validation errors, or API aborts.
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S
        )
        if response.status_code == 400 and "think" in response.text.lower():
            payload.pop("think")
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S
            )
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as exc:
        raise ProviderError(f"Ollama host {OLLAMA_HOST} unreachable. Verify daemon.") from exc
    except requests.RequestException as exc:
        body = getattr(exc.response, "text", "")[:300]
        raise ProviderError(f"Ollama rejected {model} request: {exc}. {body}") from exc


def _ollama(prompt: str, schema: type[T], model: str, temperature: float,
            require_all: bool = True, require: Optional[set] = None) -> T:
    """Handles prompt compilation, exact inference, and truncation logic."""
    compiled = _drop_unsupported(_inline_refs(schema.model_json_schema()))
    if require:
        compiled = require_fields(compiled, require)
    elif require_all:
        compiled = _require_all(compiled)

    full_prompt = prompt + _schema_instructions(compiled)
    num_ctx = _context_tokens(full_prompt)

    if int(len(full_prompt) / CHARS_PER_TOKEN) + 2048 > OLLAMA_CTX_MAX:
        raise ContextTooLarge(
            f"Prompt exceeds absolute context bounds ({OLLAMA_CTX_MAX:,} tokens). "
            "Execution blocked to prevent artificial absence manufacturing."
        )

    log.debug("ollama: %d chars -> num_ctx=%d", len(full_prompt), num_ctx)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "format": compiled,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "think": LLM_THINK,
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": OLLAMA_NUM_PREDICT},
    }

    body = _execute_request(payload, model)

    if _was_truncated(body, num_ctx):
        seen = body.get("prompt_eval_count", 0)
        if num_ctx >= OLLAMA_CTX_MAX:
            raise ContextTooLarge("Prompt context strictly exhausted.")
        
        # Escalate and retry context exhaustion up to the maximal bound.
        log.warning("Truncated at %d tokens (num_ctx=%d); retrying maximal bound.", seen, num_ctx)
        payload["options"]["num_ctx"] = min(num_ctx * 2, OLLAMA_CTX_MAX)
        body = _execute_request(payload, model)
        
        if _was_truncated(body, payload["options"]["num_ctx"]):
            raise ContextTooLarge(f"Unrecoverable truncation at num_ctx={payload['options']['num_ctx']}.")

    content = (body.get("message") or {}).get("content", "")
    if not content.strip():
        raise ProviderError(f"{model} emitted empty response.")
        
    try:
        return schema(**json.loads(content))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProviderError(f"Model {model} violated schema constraint: {exc}") from exc


# =============================================================================
#                       ********* PUBLIC API *********                       
#    Agent-facing interfaces for schema-constrained generative inference.    
# =============================================================================


def agent_model(provider: str | None = None, agent: str | None = None) -> tuple[str, str]:
    """Resolves target (provider, model) pair for the requested agent profile."""
    provider = (provider
                or (AGENT_PROVIDERS.get((agent or "").lower()) if agent else None)
                or AGENT_PROVIDER).lower()
    if AGENT_MODEL:
        return provider, AGENT_MODEL
    if agent and AGENT_MODELS.get(agent.lower()):
        return provider, AGENT_MODELS[agent.lower()]
    return provider, OLLAMA_MODEL


def structured(
    prompt: str,
    schema: type[T],
    *,
    agent: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = LLM_TEMPERATURE,
    require_all: bool = True,
    require_fields: Optional[set] = None,
) -> tuple[T, str]:
    """Executes a constrained prompt and returns a validated Pydantic instance.
    
    Args:
        prompt: The input prompt string.
        schema: The Pydantic model class defining the output structure.
        agent: Optional agent identifier determining the underlying model.
        provider: Optional provider override.
        model: Optional model override.
        temperature: Sampling temperature (defaults to LLM_TEMPERATURE).
        require_all: If True, forces all fields to be generated. Must be False for 
                     massive schemas (e.g., 118-field Extractions) to prevent context 
                     exhaustion from null-field generation.
        require_fields: Optional set of strictly required fields.
        
    Returns:
        tuple[T, str]: The parsed instance and the model identifier.
    """
    provider_resolved, default_model = agent_model(provider, agent)
    model_resolved = model or default_model

    if provider_resolved != "ollama":
        raise ProviderError(f"Unsupported local provider configuration: {provider_resolved!r}")

    result = _ollama(prompt, schema, model_resolved, temperature, require_all, require_fields)
    return result, f"ollama/{model_resolved}"


def health() -> dict[str, Any]:
    """Reports configuration readiness and target reachability status."""
    provider_resolved, model_resolved = agent_model()
    info: dict[str, Any] = {"provider": provider_resolved, "model": model_resolved, "host": OLLAMA_HOST}
    try:
        version = requests.get(f"{OLLAMA_HOST}/api/version", timeout=5).json()
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json()
        installed = [m["name"] for m in tags.get("models", [])]
        info |= {
            "reachable": True,
            "ollama_version": version.get("version"),
            "installed": installed,
            "model_present": any(m.split(":")[0] == model_resolved.split(":")[0]
                                 for m in installed),
        }
    except Exception as exc:                                      # noqa: BLE001
        info |= {"reachable": False, "error": f"Daemon unreachable: {exc}"}
    return info
