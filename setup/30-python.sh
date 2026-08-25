#!/usr/bin/env bash
# =============================================================================
#                       ********* SETUP ROUTINE *********                      
#            Infrastructure provisioning and dependency resolution.            
# =============================================================================
# Step 3 — the Python environment, inside the repository.
#
# .venv lives at the repository root (the Makefile's PY points there). The original
# layout kept it one level up, in a workspace directory that does not exist on a fresh
# VM; that is the single most common reason a copy of this project fails to start.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

say "python environment"

PYBIN="$(command -v python3.12 || command -v python3.11 || command -v python3)"
PYV="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
info "interpreter: $PYBIN ($PYV)"
[ "$(printf '%s\n3.11\n' "$PYV" | sort -V | head -1)" = "3.11" ] \
    || die "need python >= 3.11, found $PYV (run setup/00-system.sh first)"

if [ -x "$VENV/bin/python" ]; then
    ok "venv exists at $VENV"
else
    "$PYBIN" -m venv "$VENV" || die "could not create the venv"
    ok "created $VENV"
fi

"$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel \
    >"$SETUP_LOG_DIR/pip.log" 2>&1

# Pull `verify` (ASE driver) and `dev` (pytest/ruff toolchain).
# can run the test suite. Tier 1 MLIP screening ([screen]: mace-torch, chgnet, matgl) is
# NOT installed: it is triage-only, never used for a verdict, and pulls a PyTorch tree
# an order of magnitude larger than everything else here. Add it by hand if wanted.
info "installing acv + [verify,dev] (editable)"
"$VENV/bin/python" -m pip install -q -e "$ACV_ROOT[verify,dev]" \
    >>"$SETUP_LOG_DIR/pip.log" 2>&1 \
    || die "pip install failed; see $SETUP_LOG_DIR/pip.log"
ok "installed"

# `import importlib` does NOT bind the `util` submodule -- it has to be imported by
# name, or this raises AttributeError and set -e kills the step before .env is written.
"$VENV/bin/python" - <<'PY'
import importlib.util
mods = ["acv", "requests", "yaml", "pydantic", "dotenv", "matplotlib", "ase"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("   imports:", "all ok" if not missing else f"MISSING {missing}")
raise SystemExit(1 if missing else 0)
PY

# =============================================================================
#                           ********* .ENV *********                           
#            Resolve explicit system paths for micromamba overrides.           
# =============================================================================
ENV_FILE="$ACV_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    ok ".env already exists -- left alone"
else
    {
        echo "# Written by setup/30-python.sh on $(date -Is). Safe to edit."
        echo "OPENALEX_MAILTO=${OPENALEX_MAILTO:-you@example.ac.uk}"
        echo "OLLAMA_HOST=http://localhost:11434"
        echo "ACV_OLLAMA_MODEL=$MODEL"
        echo "ACV_EXTRACT_MODEL=$MODEL"
        echo "ACV_OPT_PREFIX=$OPT_PREFIX"
        echo "ACV_SIESTA_BIN=$SIESTA_ENV/bin/siesta"
        [ -x "$SIESTA_ENV/bin/mpirun" ] && echo "ACV_MPIRUN=$SIESTA_ENV/bin/mpirun"
        echo "ACV_LOCAL_RANKS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))"
    } > "$ENV_FILE"
    ok "wrote $ENV_FILE"
    sed 's/^/     /' "$ENV_FILE"
fi
