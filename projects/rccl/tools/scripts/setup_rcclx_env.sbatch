#!/bin/bash
# =============================================================================
# setup_rcclx_env.sbatch
#
# Slurm batch job that builds the RCCLX shared library on a single node:
#   1. Installs Miniconda (to a shared / NFS path) if not already present
#   2. Creates the `rcclx` conda env (Python 3.12) from conda-forge with just
#      the build tools the source build needs (cmake / ninja / perl)
#   3. Builds the RCCLX shared library (librccl.so) from source -- NOT the
#      torchcomms python package; the .so is meant to be attached to other
#      workloads (LD_LIBRARY_PATH / LD_PRELOAD) for regression testing
#   4. Verifies the librccl.so artifact and prints how to attach it
#
# The script is IDEMPOTENT: re-running skips steps that are already done.
# Everything lives under $PREFIX_ROOT so it persists on NFS and is reusable.
#
# Submit:   sbatch setup_rcclx_env.sbatch
# Or run on an interactive allocation:
#           salloc -N 1 --ntasks-per-node=8 -p meta64 -t 04:00:00 --qos=urgent
#           bash setup_rcclx_env.sbatch
# =============================================================================
#SBATCH --job-name=rcclx-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=meta64
#SBATCH --time=04:00:00
#SBATCH --qos=urgent
#SBATCH --output=rcclx-setup-%j.out
#SBATCH --error=rcclx-setup-%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override by exporting before submitting, e.g. PREFIX_ROOT=...)
# -----------------------------------------------------------------------------
# Shared/NFS root that holds conda + the build tree so it persists across jobs.
PREFIX_ROOT="${PREFIX_ROOT:-$HOME/rcclx-stack}"
CONDA_ROOT="${CONDA_ROOT:-$PREFIX_ROOT/miniconda3}"
ENV_NAME="${ENV_NAME:-rcclx}"
PY_VERSION="${PY_VERSION:-3.12}"

# torchcomms / RCCLX backend (built from source against the installed ROCm).
TORCHCOMMS_REPO="${TORCHCOMMS_REPO:-https://github.com/meta-pytorch/torchcomms.git}"
TORCHCOMMS_REF="${TORCHCOMMS_REF:-main}"
TORCHCOMMS_SRC="${TORCHCOMMS_SRC:-$PREFIX_ROOT/torchcomms}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
# GPU arch (gfx942 = MI300X/MI325X, gfx950 = MI350X/MI355X). Empty = auto-detect.
GFX_ARCH="${GFX_ARCH:-}"

MINICONDA_INSTALLER_URL="${MINICONDA_INSTALLER_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Host:           $(hostname)"
log "PREFIX_ROOT:    $PREFIX_ROOT"
log "CONDA_ROOT:     $CONDA_ROOT"
log "ENV_NAME:       $ENV_NAME (python $PY_VERSION)"
mkdir -p "$PREFIX_ROOT"

# -----------------------------------------------------------------------------
# 1. Install Miniconda if missing
# -----------------------------------------------------------------------------
if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    log "Installing Miniconda to $CONDA_ROOT ..."
    installer="$PREFIX_ROOT/miniconda.sh"
    curl -fsSL -o "$installer" "$MINICONDA_INSTALLER_URL"
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
    log "Miniconda installed: $($CONDA_ROOT/bin/conda --version)"
else
    log "Miniconda already present: $($CONDA_ROOT/bin/conda --version)"
fi

# Make conda usable in this non-interactive shell.
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# -----------------------------------------------------------------------------
# 2. Create the conda env (from conda-forge to avoid defaults-channel ToS)
# -----------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Creating conda env '$ENV_NAME' (python $PY_VERSION) from conda-forge ..."
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PY_VERSION"
else
    log "Conda env '$ENV_NAME' already exists; skipping create."
fi

conda activate "$ENV_NAME"
log "Using python: $(python -V 2>&1) at $(command -v python)"

# -----------------------------------------------------------------------------
# 3. Build the RCCLX shared library (librccl.so) from source.
# -----------------------------------------------------------------------------
RCCLX_BUILD_DIR="${RCCLX_BUILD_DIR:-$TORCHCOMMS_SRC/build/rcclx}"
RCCLX_LIB_DIR="$RCCLX_BUILD_DIR/lib"
RCCLX_INCLUDE_DIR="$RCCLX_BUILD_DIR/include/rccl"
RCCLX_SO="$RCCLX_LIB_DIR/librccl.so"

if [[ -f "$RCCLX_SO" ]]; then
    log "RCCLX library already built: $RCCLX_SO"
else
    log "Installing RCCLX build prerequisites (cmake/ninja/perl/pyyaml) ..."
    conda install -y -n "$ENV_NAME" -c conda-forge --override-channels \
        cmake ninja perl pyyaml

    if [[ ! -d "$TORCHCOMMS_SRC/.git" ]]; then
        log "Cloning torchcomms ($TORCHCOMMS_REF) -> $TORCHCOMMS_SRC"
        git clone --recursive "$TORCHCOMMS_REPO" "$TORCHCOMMS_SRC"
        git -C "$TORCHCOMMS_SRC" checkout "$TORCHCOMMS_REF"
        git -C "$TORCHCOMMS_SRC" submodule update --init --recursive
    else
        log "torchcomms source already present at $TORCHCOMMS_SRC"
    fi

    # Detect this node's GPU arch; abort rather than guess a wrong one.
    if [[ -z "$GFX_ARCH" ]]; then
        GFX_ARCH="$("$ROCM_HOME/bin/rocm_agent_enumerator" 2>/dev/null | grep -m1 '^gfx[0-9a-f]\+' || true)"
        if [[ -z "$GFX_ARCH" ]]; then
            GFX_ARCH="$("$ROCM_HOME/bin/rocminfo" 2>/dev/null | grep -o -m1 'gfx[0-9a-f]\+' || true)"
        fi
    fi
    if [[ -z "$GFX_ARCH" ]]; then
        log "ERROR: could not detect local GPU arch (set GFX_ARCH=gfxXXX and re-run)."
        exit 1
    fi
    log "Building RCCLX for local arch $GFX_ARCH (ROCM_HOME=$ROCM_HOME) ..."

    # Wipe stale third-party build dirs so they reconfigure cleanly on retry.
    # CLEAN_THIRD_PARTY_LIBS="openssl boost ..." widens; CLEAN_THIRD_PARTY=1
    # nukes the whole /tmp/third-party staging tree.
    THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-/tmp/third-party}"
    if [[ "${CLEAN_THIRD_PARTY:-0}" == 1 ]]; then
        log "Cleaning entire third-party staging dir $THIRD_PARTY_DIR ..."
        rm -rf "$THIRD_PARTY_DIR"
    else
        for lib in ${CLEAN_THIRD_PARTY_LIBS:-openssl}; do
            if [[ -e "$THIRD_PARTY_DIR/$lib" ]]; then
                log "Removing stale third-party build dir $THIRD_PARTY_DIR/$lib ..."
                rm -rf "$THIRD_PARTY_DIR/$lib"
            fi
        done
    fi

    export ROCM_HOME
    export BUILDDIR="$RCCLX_BUILD_DIR"
    export NCCL_SKIP_CONDA_INSTALL=1
    ( cd "$TORCHCOMMS_SRC" && ./build_rcclx.sh --amdgpu_targets "$GFX_ARCH" )

    if [[ ! -f "$RCCLX_SO" ]]; then
        log "ERROR: build finished but $RCCLX_SO is missing."
        log "       Inspect $RCCLX_BUILD_DIR for the actual artifact layout."
        find "$RCCLX_BUILD_DIR" -name 'librccl*' 2>/dev/null | sed 's/^/         found: /' || true
        exit 1
    fi
    log "RCCLX library built: $RCCLX_SO"
fi

# -----------------------------------------------------------------------------
# 4. Verify the RCCLX library artifact
# -----------------------------------------------------------------------------
log "Verifying RCCLX library ..."
ls -l "$RCCLX_SO" || { log "ERROR: $RCCLX_SO missing"; exit 1; }
log "Shared object dependencies (ldd):"
ldd "$RCCLX_SO" || true
if command -v nm >/dev/null 2>&1; then
    nm -D --defined-only "$RCCLX_SO" 2>/dev/null | grep -iE 'nccl(GetVersion|CommInitRank)\b' \
        | sed 's/^/  symbol: /' || log "  (no nccl* symbols found via nm)"
fi

cat <<EOF

=============================================================================
 RCCLX library ready.
   conda root : $CONDA_ROOT
   env        : $ENV_NAME
   gpu arch   : $GFX_ARCH

 Artifact (attach this to your workloads):
   library    : $RCCLX_SO
   lib dir    : $RCCLX_LIB_DIR
   include    : $RCCLX_INCLUDE_DIR

 To run a workload against this RCCLX build:
   export LD_LIBRARY_PATH=$RCCLX_LIB_DIR:\$LD_LIBRARY_PATH
   # or force-load it ahead of any system rccl:
   export LD_PRELOAD=$RCCLX_SO
=============================================================================
EOF
log "Done."
