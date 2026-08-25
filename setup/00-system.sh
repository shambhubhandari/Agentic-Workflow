#!/usr/bin/env bash
# =============================================================================
#                       ********* SETUP ROUTINE *********                      
#            Infrastructure provisioning and dependency resolution.            
# =============================================================================
# Enforce minimal OS dependencies for execution environment.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

say "system packages"

if ! have apt-get; then
    warn "not a Debian/Ubuntu system -- install these by hand and re-run:"
    warn "  python3.11+ python3-venv build-essential curl git bzip2 pciutils"
    exit 0
fi

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    build-essential curl git bzip2 ca-certificates pciutils \
    python3 python3-venv python3-dev python3-pip \
    >"$SETUP_LOG_DIR/apt.log" 2>&1 || die "apt-get failed; see $SETUP_LOG_DIR/apt.log"
ok "build tools, python3-venv, curl"

# Python must be >= 3.11 (pyproject requires-python). Ubuntu 22.04 ships 3.10, which
# Fail early on Python versions lacking union syntax (`X | Y`) support.
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$(printf '%s\n3.11\n' "$PYV" | sort -V | head -1)" != "3.11" ]; then
    warn "python3 is $PYV; this project needs >= 3.11"
    if have add-apt-repository; then
        info "installing python3.11 from deadsnakes"
        $SUDO add-apt-repository -y ppa:deadsnakes/ppa >>"$SETUP_LOG_DIR/apt.log" 2>&1
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq python3.11 python3.11-venv python3.11-dev \
            >>"$SETUP_LOG_DIR/apt.log" 2>&1 || die "could not install python3.11"
        ok "python3.11 installed -- setup/30-python.sh will use it"
    else
        die "python >= 3.11 required, found $PYV, and no add-apt-repository available"
    fi
else
    ok "python $PYV"
fi

say "gpu"
VRAM="$(gpu_vram_mib)"
if [ "${VRAM:-0}" -gt 0 ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | sed 's/^/   /'
    # CUDA needs driver 550+ on this stack; on 535 Ollama falls back to Vulkan, which
    # segfaulted on the development GPU. Reported, not fixed: replacing an Azure image's
    # driver is a bigger risk than the fallback.
    DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
    if [ "${DRV:-0}" -lt 550 ]; then
        warn "driver $DRV.x is below 550 -- Ollama may fall back to Vulkan. If inference"
        warn "is absurdly slow or segfaults, that is why."
    fi
    echo "$VRAM" > "$SETUP_LOG_DIR/vram.mib"
else
    warn "no nvidia-smi -- falling back to CPU execution."
    echo 0 > "$SETUP_LOG_DIR/vram.mib"
fi
