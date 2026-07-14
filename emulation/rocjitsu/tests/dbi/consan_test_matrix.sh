#!/usr/bin/env bash
set -euo pipefail

tier="${1:-all}"
rocjitsu_build="${ROCJITSU_BUILD_DIR:-}"
iree_build="${IREE_BUILD_DIR:-}"
hip_moi_build="${HIP_MOI_BUILD_DIR:-}"
rocm_dist="${ROCM_DIST_DIR:-}"
parallel="${CTEST_PARALLEL_LEVEL:-8}"
gpu_arch="${CONSAN_GPU_ARCH:-}"
dry_run="${CONSAN_DRY_RUN:-0}"

usage() {
  printf '%s\n' \
    "usage: consan_test_matrix.sh tier0|tier1|tier2|all" \
    "" \
    "Required environment:" \
    "  ROCJITSU_BUILD_DIR  IREE_BUILD_DIR  HIP_MOI_BUILD_DIR  ROCM_DIST_DIR" \
    "Optional: CTEST_PARALLEL_LEVEL (default 8)" \
    "          CONSAN_GPU_ARCH=gfx1201|gfx950 (auto-detected by default)" \
    "          CONSAN_DRY_RUN=1 (list and validate selections without running)"
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

if [[ -z "${gpu_arch}" ]]; then
  agent_enumerator="${rocm_dist}/bin/rocm_agent_enumerator"
  if [[ ! -x "${agent_enumerator}" ]]; then
    printf 'cannot auto-detect GPU architecture: missing %s\n' "${agent_enumerator}" >&2
    exit 2
  fi
  agents="$(${agent_enumerator})"
  if grep -qx 'gfx950' <<<"${agents}"; then
    gpu_arch=gfx950
  elif grep -qx 'gfx1201' <<<"${agents}"; then
    gpu_arch=gfx1201
  else
    printf 'no supported ConSan GPU found (need gfx1201 or gfx950)\n' >&2
    exit 2
  fi
fi
case "${gpu_arch}" in
  gfx1201|gfx950) ;;
  *) printf 'unsupported CONSAN_GPU_ARCH: %s\n' "${gpu_arch}" >&2; exit 2 ;;
esac

tier1_guarded_regex() {
  case "${gpu_arch}" in
    gfx1201)
      printf '%s' '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*large_rdna4_tileandfusewmma.*_rocm_hip$'
      ;;
    gfx950)
      printf '%s' '^(iree/tests/e2e/matmul/e2e_(batch_)?matmul_cdna4_.*tileandfusemfma.*_rocm_hip|iree/tests/e2e/linalg/check_rocm_hip_softmax\.mlir|iree/tests/e2e/linalg_ext_ops/check_rocm_hip_scan_configured\.mlir)$'
      ;;
  esac
}

tier1_compat_regex() {
  case "${gpu_arch}" in
    gfx1201)
      printf '%s' '^(iree/tests/e2e/linalg/check_rocm_hip_softmax\.mlir|iree/tests/e2e/linalg_ext_ops/check_rocm_hip_scan(_configured)?\.mlir)$'
      ;;
    gfx950)
      printf '%s' ''
      ;;
  esac
}

validate_selection_count() {
  local build_dir="$1"
  local regex="$2"
  local expected="$3"
  local label="$4"
  local listing
  listing="$(ctest --test-dir "${build_dir}" -N -R "${regex}")"
  local count
  count="$(grep -c 'Test  *#' <<<"${listing}" || true)"
  if [[ "${count}" != "${expected}" ]]; then
    printf '%s\n' "${listing}"
    printf '%s selection count mismatch: expected=%s actual=%s regex=%s\n' \
      "${label}" "${expected}" "${count}" "${regex}" >&2
    return 1
  fi
  printf '%s selection validated: %s tests\n' "${label}" "${count}"
}

tier0_live_regex() {
  case "${gpu_arch}" in
    gfx1201)
      printf '%s' '^ConSan(LdsTest|InlineShadowTest|MoiHipTest)\.|^ConSanSpillHipTest\.Gfx1201'
      ;;
    gfx950)
      printf '%s' '^ConSan(SpillHipTest\.Gfx950|LdsGfx950Test\.|MoiGfx950Test\.)'
      ;;
  esac
}

run_ctest() {
  local build_dir="$1"
  local regex="$2"
  local timeout="$3"
  shift 3
  if [[ "${dry_run}" == 1 ]]; then
    local listing
    listing="$(env "$@" ctest --test-dir "${build_dir}" -N -R "${regex}")"
    local count
    count="$(grep -c 'Test  *#' <<<"${listing}" || true)"
    printf '%s\n' "${listing}"
    if [[ "${count}" == 0 ]]; then
      printf 'dry-run selection is empty: build=%s regex=%s\n' "${build_dir}" "${regex}" >&2
      return 1
    fi
    printf 'dry-run selected %s tests from %s\n' "${count}" "${build_dir}"
    return 0
  fi
  env "$@" ctest --test-dir "${build_dir}" -j "${parallel}" --timeout "${timeout}" \
    --output-on-failure -R "${regex}"
}

run_iree_profile() {
  local flavor="$1"
  local engine="$2"
  local regex="$3"
  local timeout="$4"
  local guarded="${5:-0}"
  local profile_env=(
    "HSA_TOOLS_LIB=${hook}"
    "ROCM_PATH=${rocm_dist}"
    "HIP_PATH=${rocm_dist}"
    "LD_LIBRARY_PATH=${rocm_dist}/lib"
    "RJ_CONSAN_FLAVOR=${flavor}"
  )
  if [[ -n "${engine}" ]]; then
    profile_env+=("RJ_CONSAN_MOI_ENGINE=${engine}")
  fi
  if [[ "${gpu_arch}" == gfx1201 ]]; then
    case "${engine}" in
      record_replay|sampled)
        profile_env+=("RJ_CONSAN_MAX_PATCHES=4")
        ;;
      inline_shadow)
        profile_env+=("RJ_CONSAN_MAX_PATCHES=1" "RJ_CONSAN_MOI_OWNER_SOURCE=hw_id")
        ;;
    esac
  fi
  if [[ "${guarded}" == 1 ]]; then
    profile_env+=("RJ_CONSAN_REQUIRE_PATCH=1")
    if [[ "${engine}" == record_replay || "${engine}" == sampled || \
          "${engine}" == inline_shadow ]]; then
      profile_env+=("RJ_CONSAN_MOI_REQUIRE_RECORDS=1")
    fi
  fi
  printf '\n=== %s%s ===\n' "${flavor}" "${engine:+/${engine}}"
  run_ctest "${iree_build}" "${regex}" "${timeout}" "${profile_env[@]}"
}

run_tier0() {
  local live_regex
  live_regex="$(tier0_live_regex)"
  printf '\n=== tier0: unit and focused live controls (%s) ===\n' "${gpu_arch}"
  if [[ "${dry_run}" == 1 ]]; then
    printf 'dry-run selected focused rocjitsu unit tests from %s\n' "${unit_tests}"
  else
    "${unit_tests}" \
      '--gtest_filter=ConSan.*:ConSanMoi.*:ConSanResourcePlan.*:DbiPatchPlacementPlanner.*:TrampolineBuilder.*'
  fi
  run_ctest "${rocjitsu_build}" "${live_regex}" 30
}

run_tier1() {
  local guarded_regex
  guarded_regex="$(tier1_guarded_regex)"
  local guarded_count
  case "${gpu_arch}" in
    gfx1201) guarded_count=5 ;;
    gfx950) guarded_count=10 ;;
  esac
  validate_selection_count "${iree_build}" "${guarded_regex}" "${guarded_count}" \
    "tier1 guarded ${gpu_arch} IREE"
  printf '\n=== tier1 architecture: %s ===\n' "${gpu_arch}"
  printf '\n=== tier1: independent hip-moi semantic controls ===\n'
  run_ctest "${hip_moi_build}" '.*' 120
  run_iree_profile supercollider '' "${guarded_regex}" 60 1
  run_iree_profile moi record_replay "${guarded_regex}" 60 1
  run_iree_profile moi sampled "${guarded_regex}" 60 1
  run_iree_profile moi inline_shadow "${guarded_regex}" 60 1

  local compat_regex
  compat_regex="$(tier1_compat_regex)"
  if [[ -n "${compat_regex}" ]]; then
    validate_selection_count "${iree_build}" "${compat_regex}" 3 \
      "tier1 compatibility ${gpu_arch} IREE"
    printf '\n=== tier1: scan/softmax compatibility (unguarded) ===\n'
    run_iree_profile supercollider '' "${compat_regex}" 60
    run_iree_profile moi record_replay "${compat_regex}" 60
    run_iree_profile moi sampled "${compat_regex}" 60
    run_iree_profile moi inline_shadow "${compat_regex}" 60
  fi
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
