#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run the pinned compute/buffer CTS on the RDNA3 and RDNA4 simulators.
# Usage: ROCJITSU_CORPUS_DIR=/path/to/corpus bash run-vulkan-cts.sh \
#          [all|gfx1100|gfx1201] [pytest options, e.g. --case smoke]
# Optional: ROCJITSU_SOURCE_DIR, ROCJITSU_BUILD_DIR,
#           VULKAN_CTS_WORKERS (default up to 16), VULKAN_CTS_CPU_BUDGET
#           (total RJ execution threads, default up to 64), VK_DRIVER_FILES,
#           VULKAN_CTS_ARTIFACT_ROOT (default .pytest-artifacts/vulkan).
# Adapter options: VULKAN_CTS_BINARY, VULKAN_CTS_TIMEOUT, VULKAN_CTS_DATA_DIR.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(realpath "${ROCJITSU_SOURCE_DIR:-${script_dir}/../..}")"
if [[ -n "${ROCJITSU_BUILD_DIR:-}" ]]; then
  launcher="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
else
  launcher="$(command -v rocjitsu || true)"
  launcher="${launcher:-${source_dir}/build/tools/rocjitsu/rocjitsu}"
fi
corpus_dir="${ROCJITSU_CORPUS_DIR:-${PWD}}"
selection="${1:-all}"
if (( $# )); then shift; fi
case "${selection}" in
  all) targets=(gfx1100 gfx1201) ;;
  gfx1100|gfx1201) targets=("${selection}") ;;
  *) echo "Usage: $0 [all|gfx1100|gfx1201] [pytest options]" >&2; exit 2 ;;
esac
available_cpus="$(nproc)"
default_budget=$((available_cpus < 64 ? available_cpus : 64))
cpu_budget="${VULKAN_CTS_CPU_BUDGET:-${default_budget}}"
if [[ ! "${cpu_budget}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VULKAN_CTS_CPU_BUDGET must be a positive integer" >&2
  exit 2
fi
default_workers=$((cpu_budget < 16 ? cpu_budget : 16))
workers="${VULKAN_CTS_WORKERS:-${default_workers}}"
if [[ ! "${workers}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VULKAN_CTS_WORKERS must be a positive integer" >&2
  exit 2
fi
if (( workers > cpu_budget )); then
  echo "VULKAN_CTS_CPU_BUDGET must be at least VULKAN_CTS_WORKERS" >&2
  exit 2
fi
rj_budget=$((cpu_budget / workers))
echo "CPU budget: ${workers} pytest workers, up to ${rj_budget} RJ execution threads each"
if [[ ! -x "${launcher}" ]]; then
  echo "Missing executable RocJITsu launcher: ${launcher}. Build RocJITsu or set ROCJITSU_BUILD_DIR." >&2
  exit 1
fi
launcher="$(realpath "${launcher}")"
cd "${corpus_dir}"
if [[ ! -f tests/test_suites/vulkan.py ]]; then
  echo "Missing Vulkan corpus adapter in ${PWD}. Set ROCJITSU_CORPUS_DIR to the Vulkan corpus checkout." >&2
  exit 1
fi

if [[ -z "${VK_DRIVER_FILES:-}" ]]; then
  for manifest in /usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
                  /usr/share/vulkan/icd.d/radeon_icd.json; do
    if [[ -f "${manifest}" ]]; then
      export VK_DRIVER_FILES="${manifest}"
      break
    fi
  done
fi
: "${VK_DRIVER_FILES:?Install mesa-vulkan-drivers or set VK_DRIVER_FILES to the RADV ICD manifest}"
python3 "${script_dir}/vulkan-cts-utils.py" "${VK_DRIVER_FILES}"

# Select system Mesa/libdrm instead of SDK copies. Do not allow an inherited
# physical-device selector or ICD override to redirect the simulated workload.
run_wrapper_prefix=(
  env
  -u LD_LIBRARY_PATH
  -u LD_PRELOAD
  -u VK_ICD_FILENAMES
  -u VK_ADD_DRIVER_FILES
  -u VK_INSTANCE_LAYERS
  -u VK_LOADER_DRIVERS_SELECT
  -u VK_LOADER_DRIVERS_DISABLE
  -u RADV_FORCE_FAMILY
  "VK_DRIVER_FILES=$(realpath "${VK_DRIVER_FILES}")"
  DRI_PRIME=
)
status=0
for target in "${targets[@]}"; do
  case "${target}" in
    gfx1100) config=gfx1100_w7900.json ;;
    gfx1201) config=gfx1201_r9700.json ;;
  esac
  artifact_dir="${VULKAN_CTS_ARTIFACT_ROOT:-.pytest-artifacts/vulkan}-${target}"
  mkdir -p "${artifact_dir}"
  # Divide the budget among independent processes while retaining the shipped
  # target's allocation table and engine/dispatch policy.
  printf -v wrapper '%q ' "${run_wrapper_prefix[@]}" "${launcher}" \
    --config "${source_dir}/configs/${config}" --cpu-thread-budget "${rj_budget}" --
  python3 -m pytest tests/test_corpus.py --suite vulkan --target "${target}" \
    --run-wrapper "${wrapper}" --artifact-directory "${artifact_dir}" \
    --junitxml "${artifact_dir}/junit.xml" -n "${workers}" \
    -p no:timeout -q --tb=short "$@" || status=1
done
exit "${status}"
