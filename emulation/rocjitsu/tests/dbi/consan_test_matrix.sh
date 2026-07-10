#!/usr/bin/env bash
set -euo pipefail

tier="${1:-all}"
rocjitsu_build="${ROCJITSU_BUILD_DIR:-}"
iree_build="${IREE_BUILD_DIR:-}"
hip_moi_build="${HIP_MOI_BUILD_DIR:-}"
rocm_dist="${ROCM_DIST_DIR:-}"
parallel="${CTEST_PARALLEL_LEVEL:-8}"

usage() {
  printf '%s\n' \
    "usage: consan_test_matrix.sh tier0|tier1|tier2|all" \
    "" \
    "Required environment:" \
    "  ROCJITSU_BUILD_DIR  IREE_BUILD_DIR  HIP_MOI_BUILD_DIR  ROCM_DIST_DIR" \
    "Optional: CTEST_PARALLEL_LEVEL (default 8)"
}

case "${tier}" in
  tier0|tier1|tier2|all) ;;
  *) usage >&2; exit 2 ;;
esac

for value in rocjitsu_build iree_build hip_moi_build rocm_dist; do
  if [[ -z "${!value}" ]]; then
    usage >&2
    exit 2
  fi
done

hook="${rocjitsu_build}/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
unit_tests="${rocjitsu_build}/tests/rocjitsu_tests"
for path in "${hook}" "${unit_tests}" "${iree_build}" "${hip_moi_build}" "${rocm_dist}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing required path: %s\n' "${path}" >&2
    exit 2
  fi
done

run_ctest() {
  local build_dir="$1"
  local regex="$2"
  local timeout="$3"
  shift 3
  env "$@" ctest --test-dir "${build_dir}" -j "${parallel}" --timeout "${timeout}" \
    --output-on-failure -R "${regex}"
}

run_iree_profile() {
  local flavor="$1"
  local engine="$2"
  local regex="$3"
  local timeout="$4"
  local profile_env=(
    "HSA_TOOLS_LIB=${hook}"
    "ROCM_PATH=${rocm_dist}"
    "HIP_PATH=${rocm_dist}"
    "LD_LIBRARY_PATH=${rocm_dist}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "RJ_CONSAN_FLAVOR=${flavor}"
  )
  if [[ -n "${engine}" ]]; then
    profile_env+=("RJ_CONSAN_MOI_ENGINE=${engine}")
  fi
  printf '\n=== %s%s ===\n' "${flavor}" "${engine:+/${engine}}"
  run_ctest "${iree_build}" "${regex}" "${timeout}" "${profile_env[@]}"
}

run_tier0() {
  printf '\n=== tier0: unit and focused live controls ===\n'
  "${unit_tests}" \
    '--gtest_filter=ConSan.*:ConSanMoi.*:ConSanResourcePlan.*:DbiPatchPlacementPlanner.*:TrampolineBuilder.*'
  run_ctest "${rocjitsu_build}" \
    '^ConSan(SpillHipTest|LdsTest|InlineShadowTest|MoiHipTest)\.' 30
}

run_tier1() {
  local regex='^(iree/tests/e2e/matmul/e2e_matmul_rocm_.*rdna4_tileandfusewmma.*rocm_hip|iree/tests/e2e/linalg/check_rocm_hip_softmax\.mlir|iree/tests/e2e/linalg_ext_ops/check_rocm_hip_scan(_configured)?\.mlir)$'
  printf '\n=== tier1: independent hip-moi semantic controls ===\n'
  run_ctest "${hip_moi_build}" '.*' 120
  run_iree_profile supercollider '' "${regex}" 60
  run_iree_profile moi record_replay "${regex}" 60
  run_iree_profile moi sampled "${regex}" 60
  run_iree_profile moi inline_shadow "${regex}" 60
}

run_tier2() {
  local regex='^iree/tests/e2e/.*(rocm_hip|rocm-rocm)'
  run_iree_profile supercollider '' "${regex}" 60
  run_iree_profile moi record_replay "${regex}" 60
  run_iree_profile moi sampled "${regex}" 60
  run_iree_profile moi inline_shadow "${regex}" 60
}

case "${tier}" in
  tier0) run_tier0 ;;
  tier1) run_tier1 ;;
  tier2) run_tier2 ;;
  all) run_tier0; run_tier1; run_tier2 ;;
esac

printf '\nConSan %s matrix passed.\n' "${tier}"
