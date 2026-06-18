# Environment for running the rocjitsu-corpus gfx1250 regression under
# mirage + HotSwap.
#
# Point ROCM_VENV (or ROCM_PATH) at your install, or pre-populate PATH before
# invoking the runner.

if [[ -n "${ROCM_VENV:-}" && -x "$ROCM_VENV/bin/rocm-sdk" ]]; then
  SDK_ROOT=$("$ROCM_VENV/bin/rocm-sdk" path --root)
  export ROCM_PATH="${ROCM_PATH:-$SDK_ROOT}"
  export HIP_PATH="${HIP_PATH:-$SDK_ROOT}"
  export PATH="$ROCM_VENV/bin:$SDK_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$SDK_ROOT/lib:$SDK_ROOT/lib64:$SDK_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
fi

# The corpus VMFBs are compiled for gfx1250; HotSwap retargets them onto the
# physical host GPU. mirage derives HSA_HOTSWAP_ISA_OVERRIDE from the detected
# GPU; the source target just needs to match the corpus build arch.
export HSA_HOTSWAP_SOURCE_TARGET=${HSA_HOTSWAP_SOURCE_TARGET:-gfx1250:32}

# HotSwap runs the retargeted code on real hardware. Make sure no stale rocjitsu
# interposer or sim config leaks in from a previous shell.
unset LD_PRELOAD
unset HSA_MODEL_LIB
unset HSA_MODEL_TOPOLOGY
unset HSA_OVERRIDE_GFX_VERSION
