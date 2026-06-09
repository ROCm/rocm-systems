# Environment for running the rocjitsu-corpus gfx1250 regression under
# mirage + rocjitsu (the software GPU emulator).
#
# Unlike the corpus's stock kmd.so env, this file does NOT set LD_PRELOAD or
# RJ_CONFIG by hand: mirage injects the rocjitsu kmd.so interposer and generates
# the sim config itself when it runs each IREE tool via
# `mirage run --profile <profile> -- <tool> ...`. This file's only job is to make
# the ROCm runtime and the IREE tools discoverable in the outer shell that
# drives the corpus runner.
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

# Let mirage own the emulator wiring; clear anything a previous shell may have
# left so the rocjitsu injection is the single source of truth.
unset LD_PRELOAD
unset RJ_CONFIG
unset HSA_MODEL_LIB
unset HSA_MODEL_TOPOLOGY
unset HSA_OVERRIDE_GFX_VERSION
export ROCPROFILER_REGISTER_ENABLED=0
