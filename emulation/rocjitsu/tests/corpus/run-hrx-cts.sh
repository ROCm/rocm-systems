#!/usr/bin/env bash
# Run the pinned HRX AMDGPU CTS on the gfx1201 and single-XCC gfx1250 simulators.
# Usage: ROCJITSU_CORPUS_DIR=/path/to/corpus bash run-hrx-cts.sh [all|gfx1201|gfx1250] [pytest options]
# Optional: ROCJITSU_SOURCE_DIR, ROCJITSU_BUILD_DIR, HRX_CTS_BUILD_ROOT.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(realpath "${ROCJITSU_SOURCE_DIR:-${script_dir}/../..}")"
if [[ -n "${ROCJITSU_BUILD_DIR:-}" ]]; then
  launcher="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
else
  launcher="$(command -v rocjitsu || true)"
  launcher="${launcher:-${source_dir}/build/tools/rocjitsu/rocjitsu}"
fi
corpus_dir="$(realpath "${ROCJITSU_CORPUS_DIR:-${PWD}}")"
build_root="${HRX_CTS_BUILD_ROOT:-${corpus_dir}/.build/hrx-system}"
selection="${1:-all}"
if (( $# )); then shift; fi
case "$selection" in
  all) targets=(gfx1201 gfx1250) ;;
  gfx1201|gfx1250) targets=("$selection") ;;
  *) echo "Usage: $0 [all|gfx1201|gfx1250] [pytest options]" >&2; exit 2 ;;
esac
if [[ ! -x "$launcher" ]]; then
  echo "Missing RocJITsu launcher: $launcher" >&2
  exit 1
fi
launcher="$(realpath "$launcher")"
cd "$corpus_dir"
if [[ ! -f tests/test_suites/hrx.py ]]; then
  echo "Missing HRX corpus adapter: $corpus_dir" >&2
  exit 1
fi

rocm_root="$(rocm-sdk path --root)"
export LD_LIBRARY_PATH="$rocm_root/lib:${LD_LIBRARY_PATH:-}"
export IREE_HAL_AMDGPU_LIBHSA_PATH="$rocm_root/lib/libhsa-runtime64.so.1"
status=0
for target in "${targets[@]}"; do
  config="${source_dir}/configs/gfx1201_r9700.json"
  if [[ "$target" == gfx1250 ]]; then
    config="${source_dir}/configs/gfx1250_mi455x_single_xcc.json"
  fi
  artifact_dir=".pytest-artifacts/hrx-${target}"
  mkdir -p "$artifact_dir"
  printf -v wrapper '%q ' "$launcher" --config "$config" --
  HRX_SYSTEM_BUILD_DIR="${HRX_SYSTEM_BUILD_DIR:-${build_root}/build/${target}}" \
    python3 -m pytest tests/test_corpus.py --suite hrx --target "$target" \
      --run-wrapper "$wrapper" --artifact-directory "$artifact_dir" \
      --junitxml "$artifact_dir/junit.xml" -q --tb=short "$@" || status=1
done
exit "$status"
