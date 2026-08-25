"""Configuration, environment resolution, and static path definitions.

Resolves all pipeline tunables through a strict precedence hierarchy: 
Environment Overrides > YAML Configurations > Static Code Defaults. 
Provides unified access to physical paths and architectural limits.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# =============================================================================
#                   ********* ROOTS & ENVIRONMENT *********                  
#         Absolute path boundaries and local environment overrides.          
# =============================================================================

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = PROJECT_ROOT.parent

load_dotenv(REPO_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=True)

CONFIGS = PROJECT_ROOT / "configs"

# =============================================================================
#                 ********* CONFIGURATION LOADERS *********                  
#     YAML parsers with strict fail-fast integrity against silent defaults.  
# =============================================================================

def _config(name: str) -> dict[str, Any]:
    """Loads a YAML configuration map, returning an empty dict if absent.
    
    Raises a RuntimeError if the file exists but cannot be parsed or read, 
    preventing silent fallback to default values and false configuration attribution.
    """
    path = CONFIGS / f"{name}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"PyYAML is required to read configs/{name}.yaml") from exc
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        raise RuntimeError(f"configs/{name}.yaml is not valid YAML: {exc}") from exc


def config_fingerprint() -> str:
    """Computes a SHA-256 digest over all active configuration files."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(CONFIGS.glob("*.yaml")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def effective(section: str, key: str, passed: Any = None, default: Any = None) -> Any:
    """Resolves a configuration value respecting caller > section > default hierarchy."""
    if passed is not None:
        return passed
    mapping = _SECTIONS.get(section, {})
    parts = key.split(".")
    for p in parts[:-1]:
        mapping = mapping.get(p, {})
        if not isinstance(mapping, dict):
            return default
    return mapping.get(parts[-1], default)


def _setting(env_var: str, yaml_key: str, default: Any, cast: Callable = str, section: str = "models") -> Any:
    """Resolves a setting respecting env_var > yaml_key > default hierarchy."""
    val = os.getenv(env_var)
    if val is not None:
        return cast(val)
    val = effective(section, yaml_key)
    if val is not None:
        return cast(val)
    return cast(default) if default is not None else None


_MODELS = _config("models")
_CORPUS = _config("corpus")
_TIER0 = _config("tier0")
_TIER2 = _config("tier2")

_SECTIONS = {"corpus": _CORPUS, "tier0": _TIER0, "tier2": _TIER2, "models": _MODELS}


# =============================================================================
#                 ********* PUBLISHED ARTEFACTS *********                 
#    Static paths defining the structure of the published data repository.   
# =============================================================================

DATA = PROJECT_ROOT / "data"
EXTERNAL = DATA / "external"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FULLTEXT = RAW / "fulltext"
GENERATED = DATA / "generated"
FIGURES = PROJECT_ROOT / "figures"
PROMPTS = PACKAGE_ROOT / "prompts"

for _d in (RAW, INTERIM, PROCESSED, FULLTEXT, GENERATED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

CORPUS_LOCKED = RAW / "corpus" / "corpus_298_locked.jsonl"
FULLTEXT_MANIFEST = RAW / "fulltext_manifest.tsv"
REPORTED_VALUES = RAW / "reported_values.yaml"
RUNTIME_LOGS = RAW / "runtime_logs"
LABELS_EXPERT_RAW = RAW / "labels_expert"
LABELS_MODEL = RAW / "labels_model"

LABELS_EXPERT = INTERIM / "labels_expert_merged.csv"
EXTRACTION = INTERIM / "extraction"
CONFIGURATIONS = ("rtx3050_q4_0", "tesla_t4_q4_0", "tesla_t4_f16")

EXTRACTION_SUMMARY = PROCESSED / "extraction_summary.json"
EVALUATION = PROCESSED / "evaluation.json"
REPORTABILITY = PROCESSED / "reportability.jsonl"
VERIFICATION = PROCESSED / "verification"
PARITY_POINTS = PROCESSED / "parity_points.jsonl"
NUMBERS_TEX = PROCESSED / "numbers.tex"

def union(configuration: str, gated: bool = True) -> Path:
    """Locates the extraction union for a specific hardware configuration."""
    return EXTRACTION / configuration / ("union_gated.jsonl" if gated else "union.jsonl")

def summary(configuration: str = "rtx3050_q4_0") -> Path:
    """Locates the reportability summary for a specific hardware configuration."""
    if configuration == "rtx3050_q4_0":
        return EXTRACTION_SUMMARY
    return PROCESSED / f"extraction_summary_{configuration}.json"


OPENALEX_MAILTO = _setting("OPENALEX_MAILTO", "search.polite_pool_mailto", "acv-research@ucl.ac.uk", section="corpus")
OPENALEX_BASE = "https://api.openalex.org"

# =============================================================================
#                   ********* LANGUAGE MODEL *********                   
#      Local Open-Weights resolution, agent maps, and context windows.       
# =============================================================================

EXTRACT_MODEL = _setting("ACV_EXTRACT_MODEL", "default_model", "qwen3:4b")
AGENT_PROVIDER = _setting("ACV_AGENT_PROVIDER", "provider", "ollama")
AGENT_MODEL = os.getenv("ACV_AGENT_MODEL", "")

AGENT_MODELS: dict[str, str] = dict(_MODELS.get("agents") or {})

LLM_TEMPERATURE = _setting("ACV_LLM_TEMPERATURE", "temperature", 0.0, float)
LLM_THINK = bool(_MODELS.get("think", False))
LLM_NUM_PREDICT = _setting("ACV_LLM_NUM_PREDICT", "num_predict", 6144, int)
LLM_CTX_MIN = _setting("ACV_LLM_CTX_MIN", "context.min", 4096, int)
LLM_CTX_MAX = _setting("ACV_LLM_CTX_MAX", "context.max", 32768, int)
LLM_CHARS_PER_TOKEN = _setting("ACV_LLM_CHARS_PER_TOKEN", "context.chars_per_token", 3.5, float)
LLM_CTX_HEADROOM = _setting("ACV_LLM_CTX_HEADROOM", "context.headroom_over_num_predict", 512, int)
LLM_CTX_BUCKET = _setting("ACV_LLM_CTX_BUCKET", "context.bucket", 4096, int)
LLM_KEEP_ALIVE = _setting("ACV_LLM_KEEP_ALIVE", "keep_alive", "5m")
LLM_TIMEOUT_S = _setting("ACV_LLM_TIMEOUT_S", "timeout_s", 3600, int)

AGENT_PROVIDERS: dict[str, str] = {}
for _pair in os.getenv("ACV_AGENT_PROVIDERS", "").split(","):
    if "=" in _pair:
        _agent, _prov = _pair.split("=", 1)
        AGENT_PROVIDERS[_agent.strip().lower()] = _prov.strip().lower()

OLLAMA_HOST = _setting("OLLAMA_HOST", "host", "http://localhost:11434")
OLLAMA_MODEL = _setting("ACV_OLLAMA_MODEL", "default_model", "qwen3:4b")


# =============================================================================
#                     ********* SIESTA & MPI *********                     
#           Physical execution bounds and parallelization limits.            
# =============================================================================

def _opt_roots() -> list[Path]:
    """Generates deduplicated candidate prefixes for locally installed toolchains."""
    roots = [Path(os.getenv("ACV_OPT_PREFIX") or (Path.home() / "opt"))]
    roots.append(Path.home() / "opt")
    for anchor in (PROJECT_ROOT, REPO_ROOT):
        roots.append(anchor / "opt")
        if len(anchor.parents) > 1:
            roots.append(anchor.parents[1] / "opt")
    seen = set()
    out = []
    for r in roots:
        if str(r) not in seen:
            seen.add(str(r))
            out.append(r)
    return out

SIESTA_CANDIDATES = [
    Path(p) for p in filter(None, [
        os.getenv("ACV_SIESTA_BIN"),
        *[str(r / env / "bin" / "siesta")
          for r in _opt_roots() for env in ("siesta-mpi", "siesta-env")],
        shutil.which("siesta"),
        "/usr/local/bin/siesta",
        "/usr/bin/siesta",
    ])
]

PSEUDO_DIR = EXTERNAL / "pseudopotentials"

EXECUTOR = os.getenv("ACV_EXECUTOR", "local")

LOCAL_MPI_RANKS = int(os.getenv("ACV_LOCAL_RANKS") or max(1, (os.cpu_count() or 2) - 1))
MPIRUN = (os.getenv("ACV_MPIRUN")
          or next((str(r / "siesta-mpi" / "bin" / "mpirun") for r in _opt_roots()
                   if (r / "siesta-mpi" / "bin" / "mpirun").exists()), None)
          or shutil.which("mpirun")
          or "mpirun")
LOCAL_MIN_FREE_RAM_GB = float(os.getenv("ACV_MIN_FREE_RAM_GB", "2.0"))
LOCAL_MIN_FREE_DISK_GB = float(os.getenv("ACV_MIN_FREE_DISK_GB", "10.0"))
