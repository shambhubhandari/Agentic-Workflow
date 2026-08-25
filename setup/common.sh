#!/usr/bin/env bash
# Shared helpers. Sourced by every numbered step; not executable on its own.

# =============================================================================
#                       ********* SETUP ROUTINE *********                      
#            Infrastructure provisioning and dependency resolution.            
# =============================================================================

set -euo pipefail

ACV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACV_ROOT
export OPT_PREFIX="${ACV_OPT_PREFIX:-$HOME/opt}"
export MAMBA_ROOT="${MAMBA_ROOT:-$OPT_PREFIX/micromamba}"
export SIESTA_ENV="${SIESTA_ENV:-$OPT_PREFIX/siesta-mpi}"
export VENV="$ACV_ROOT/.venv"
export MODEL="${ACV_MODEL:-qwen3:4b}"
export SETUP_LOG_DIR="$ACV_ROOT/logs/setup"

mkdir -p "$SETUP_LOG_DIR"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33mwarn\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[31mFAILED\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Ensure idempotency for all setup steps.
# `skip_if <test-command> <message>` returns 0 (meaning "already done") when the test
# passes, and the caller returns early.
skip_if() {
    if eval "$1" >/dev/null 2>&1; then ok "$2"; return 0; fi
    return 1
}

# Total VRAM in MiB of the first GPU, or 0 when there is no nvidia-smi.
gpu_vram_mib() {
    if have nvidia-smi; then
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d '[:space:]' || echo 0
    else
        echo 0
    fi
}
