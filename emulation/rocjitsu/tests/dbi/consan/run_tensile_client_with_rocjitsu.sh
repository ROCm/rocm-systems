#!/usr/bin/env bash
set -euo pipefail

: "${ROCJITSU_EXE:?set ROCJITSU_EXE to the RocJITsu launcher}"
: "${ROCJITSU_CONFIG:?set ROCJITSU_CONFIG to the simulator JSON}"
: "${TENSILE_CLIENT_EXE:?set TENSILE_CLIENT_EXE to tensilelite-client}"

[[ -x "$ROCJITSU_EXE" ]] || {
  echo "error: RocJITsu launcher is not executable: $ROCJITSU_EXE" >&2
  exit 2
}
[[ -r "$ROCJITSU_CONFIG" ]] || {
  echo "error: RocJITsu config is not readable: $ROCJITSU_CONFIG" >&2
  exit 2
}
[[ -x "$TENSILE_CLIENT_EXE" ]] || {
  echo "error: Tensile client is not executable: $TENSILE_CLIENT_EXE" >&2
  exit 2
}

unset LD_PRELOAD
unset RJ_CONFIG
unset HSA_MODEL_LIB
unset HSA_MODEL_TOPOLOGY
unset HSA_OVERRIDE_GFX_VERSION

exec "$ROCJITSU_EXE" --config "$ROCJITSU_CONFIG" -- "$TENSILE_CLIENT_EXE" "$@"
