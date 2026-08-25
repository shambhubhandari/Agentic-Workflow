#!/usr/bin/env bash
# =============================================================================
#                       ********* SETUP ROUTINE *********                      
#            Infrastructure provisioning and dependency resolution.            
# =============================================================================
# Step 2 — SIESTA, via micromamba and conda-forge.
#
# Resolve SIESTA and MPI dependencies via isolated micromamba environment.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

say "siesta"

if [ -x "$SIESTA_ENV/bin/siesta" ]; then
    ok "already present at $SIESTA_ENV/bin/siesta"
else
    # --- micromamba --------------------------------------------------------------
    MAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
    if [ ! -x "$MAMBA_BIN" ]; then
        info "installing micromamba to $MAMBA_ROOT"
        mkdir -p "$MAMBA_ROOT/bin"
        curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xvj -C "$MAMBA_ROOT" --strip-components=1 bin/micromamba \
            >"$SETUP_LOG_DIR/micromamba.log" 2>&1 \
            || die "micromamba download failed; see $SETUP_LOG_DIR/micromamba.log"
        [ -x "$MAMBA_ROOT/micromamba" ] && mv "$MAMBA_ROOT/micromamba" "$MAMBA_BIN"
        chmod +x "$MAMBA_BIN"
    fi
    ok "micromamba $("$MAMBA_BIN" --version 2>/dev/null)"

    # --- the environment ---------------------------------------------------------
    # Enforce openmpi variant to prevent 2.4x serial-build latency penalty.
    info "solving siesta + openmpi from conda-forge"
    export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
    if ! "$MAMBA_BIN" create -y -p "$SIESTA_ENV" -c conda-forge \
            'siesta=*=*openmpi*' openmpi \
            >"$SETUP_LOG_DIR/siesta-solve.log" 2>&1; then
        warn "falling back to default siesta build"
        "$MAMBA_BIN" create -y -p "$SIESTA_ENV" -c conda-forge siesta openmpi \
            >>"$SETUP_LOG_DIR/siesta-solve.log" 2>&1 \
            || die "could not install siesta; see $SETUP_LOG_DIR/siesta-solve.log"
    fi
    [ -x "$SIESTA_ENV/bin/siesta" ] || die "siesta binary missing after install"
    ok "installed to $SIESTA_ENV"
fi

# =============================================================================
#                       ********* PROVE IT RUNS *********                      
#            Enforce binary linkage constraints prior to execution.            
# =============================================================================
say "checking the binary"
VER="$("$SIESTA_ENV/bin/siesta" --version 2>&1 | head -2 | tr '\n' ' ' || true)"
if [ -z "$VER" ]; then
    printf 'SystemName probe\nSystemLabel probe\n' > /tmp/acv-siesta-probe.fdf
    ( cd /tmp && timeout 60 "$SIESTA_ENV/bin/siesta" < acv-siesta-probe.fdf >/dev/null 2>&1 ) \
        || warn "siesta did not answer --version and the probe run also failed"
fi
info "siesta: ${VER:-<no --version output; probed instead>}"
ldd "$SIESTA_ENV/bin/siesta" 2>/dev/null | grep -i "not found" \
    && die "siesta has unresolved shared libraries" || ok "shared libraries resolve"

if [ -x "$SIESTA_ENV/bin/mpirun" ]; then
    ok "mpirun at $SIESTA_ENV/bin/mpirun"
else
    warn "no mpirun in the environment -- calculations will run on a single core"
fi

# =============================================================================
#                     ********* PSEUDOPOTENTIALS *********                     
#   Verify bundled pseudopotential presence to enforce execution consistency.  
# =============================================================================
N_PSEUDO="$(find "$ACV_ROOT/data/raw/pseudos" -name '*.psml' 2>/dev/null | wc -l)"
[ "$N_PSEUDO" -gt 0 ] && ok "$N_PSEUDO pseudopotentials shipped with the repository" \
    || warn "data/raw/pseudos is empty -- SIESTA will fail at the first calculation"
