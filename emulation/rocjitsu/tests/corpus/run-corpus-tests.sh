#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Run the ROCjitsu pytest corpus under simulated GPU targets.
#
# Usage:
#   ROCM_PATH=<rocm-root> ROCJITSU_SOURCE_DIR=<rocjitsu-source> \
#     ./tests/corpus/run-corpus-tests.sh [options]
#
# Options:
#   --workers N          Number of pytest-xdist workers (default: 8)
#   --soft-timeout N     Per-test timeout for the first run (default: 30)
#   --hard-timeout N     Per-test timeout for failed-test reruns (default: 60)
#   --rerun-timeout N    Overall failed-test rerun budget (default: 1200)
#   --sanitizer MODE     Launcher instrumentation: none, clang-asan, or gcc-asan
#   --rerun-failed       Rerun only tests that failed the soft-timeout pass
#   --warn-perf          Warn about passing tests close to the soft timeout
#
# Environment variables:
#   ROCM_PATH            Required ROCm installation root
#   ROCJITSU_SOURCE_DIR  Required rocjitsu source directory
#
# Outputs:
#   .pytest-artifacts/<target>/           Corpus harness logs and artifacts
#   .pytest-artifacts/junit/<target>.xml  Soft-timeout JUnit report per target
#   .pytest-cache/<target>/               Pytest cache with the lastfailed list

set -euo pipefail

: "${ROCM_PATH:?ROCM_PATH must be set}"
: "${ROCJITSU_SOURCE_DIR:?ROCJITSU_SOURCE_DIR must be set}"

worker_count=8
soft_timeout_seconds=30
hard_timeout_seconds=60
rerun_timeout_seconds=1200
rerun_failed=false
warn_perf=false
sanitizer_mode=none

usage() {
  echo "Usage: $0 [--workers N] [--soft-timeout N] [--hard-timeout N] [--rerun-timeout N]" \
    "[--sanitizer none|clang-asan|gcc-asan] [--rerun-failed] [--warn-perf]" >&2
}

targets=(
  "gfx942 gfx942_cdna3.json gfx942_skip_tests.json"
  "gfx950 gfx950_mi355x.json gfx950_skip_tests.json"
  "gfx1100 gfx1100_w7900.json gfx1100_skip_tests.json"
  "gfx1201 gfx1201_r9700.json gfx1201_skip_tests.json"
  "gfx1250 gfx1250_mi455x.json gfx1250_skip_tests.json"
)

while (( $# )); do
  case "$1" in
    --workers)
      if (( $# < 2 )); then
        echo "--workers requires a value" >&2
        usage
        exit 1
      fi
      worker_count="$2"
      shift 2
      ;;
    --soft-timeout)
      if (( $# < 2 )); then
        echo "--soft-timeout requires a value" >&2
        usage
        exit 1
      fi
      soft_timeout_seconds="$2"
      shift 2
      ;;
    --hard-timeout)
      if (( $# < 2 )); then
        echo "--hard-timeout requires a value" >&2
        usage
        exit 1
      fi
      hard_timeout_seconds="$2"
      shift 2
      ;;
    --rerun-timeout)
      if (( $# < 2 )); then
        echo "--rerun-timeout requires a value" >&2
        usage
        exit 1
      fi
      rerun_timeout_seconds="$2"
      shift 2
      ;;
    --sanitizer)
      if (( $# < 2 )); then
        echo "--sanitizer requires a mode" >&2
        usage
        exit 1
      fi
      sanitizer_mode="$2"
      case "${sanitizer_mode}" in
        none|clang-asan|gcc-asan) ;;
        *)
          echo "Unknown sanitizer mode: ${sanitizer_mode}" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --rerun-failed)
      rerun_failed=true
      shift
      ;;
    --warn-perf)
      warn_perf=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

numeric_options=(
  "worker_count:--workers"
  "soft_timeout_seconds:--soft-timeout"
  "hard_timeout_seconds:--hard-timeout"
  "rerun_timeout_seconds:--rerun-timeout"
)
for numeric_option in "${numeric_options[@]}"; do
  numeric_name="${numeric_option%%:*}"
  numeric_flag="${numeric_option#*:}"
  numeric_value="${!numeric_name}"
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${numeric_flag} requires a positive integer" >&2
    exit 1
  fi
done

corpus_test_status=0
corpus_work_dir="$(pwd -P)"
junit_dir="${corpus_work_dir}/.pytest-artifacts/junit"
junit_xml_paths=()

# A report left by an earlier run would otherwise be reported as this run's.
rm -rf "${junit_dir}"

# Direct simulator tests must bypass ROCr's built-in translation so every lane,
# including release, executes the requested architecture semantics unchanged.
run_wrapper_prefix=(
  env
  -u LD_PRELOAD
  -u HSA_HOTSWAP_ENABLE
  "HSA_HOTSWAP_DISABLE=1"
)

if ! rocjitsu_launcher="$(command -v rocjitsu)"; then
  echo "Could not resolve rocjitsu on PATH for corpus tests" >&2
  exit 1
fi
corpus_process_supervisor="${ROCJITSU_SOURCE_DIR}/tests/corpus/corpus-process-supervisor.sh"
if ! command -v setpriv >/dev/null || ! command -v setsid >/dev/null ||
   ! command -v timeout >/dev/null ||
   [[ ! -x "${corpus_process_supervisor}" ]]; then
  echo "Could not resolve the corpus process cleanup tools" >&2
  exit 1
fi

if [[ "${sanitizer_mode}" != none ]]; then
  asan_symbolizer="${ROCM_PATH}/lib/llvm/bin/llvm-symbolizer"
  if [[ ! -x "${asan_symbolizer}" ]]; then
    echo "Could not resolve the ASan symbolizer for corpus tests" >&2
    exit 1
  fi
  # The corpus target groups mix HIP-backed and pure simulator programs, so the
  # wrapper cannot vary this setting per case. LeakSanitizer's stop-the-world
  # scan stalls on the multi-gigabyte HIP mappings; ASan and UBSan remain active.
  corpus_asan_options="${ASAN_OPTIONS:+${ASAN_OPTIONS}:}detect_leaks=0"
  run_wrapper_prefix+=(
    "ASAN_OPTIONS=${corpus_asan_options}"
    "ASAN_SYMBOLIZER_PATH=${asan_symbolizer}"
  )
fi

# Keep HIP startup ordering local to the corpus harness. The helper runs inside
# rocjitsu's launched subtree, appends HIP after ASan and the interposer, and
# replaces itself with the suite command.
child_command_prefix=()
if [[ "${sanitizer_mode}" == clang-asan ]]; then
  hip_runtime="${ROCM_PATH}/lib/libamdhip64.so"
  if [[ ! -f "${hip_runtime}" ]]; then
    echo "Could not resolve HIP runtime for Clang ASan corpus preload" >&2
    exit 1
  fi
  corpus_hip_preload="${ROCJITSU_SOURCE_DIR}/tests/corpus/corpus-hip-preload.sh"
  if [[ ! -f "${corpus_hip_preload}" ]]; then
    echo "Could not resolve the corpus HIP preload helper" >&2
    exit 1
  fi
  child_command_prefix=(bash "${corpus_hip_preload}" "${hip_runtime}")

  # Fail once before the target loop if this launcher is not an ASan build or
  # does not construct the expected shared-runtime/interposer preload order.
  preflight_config="${ROCJITSU_SOURCE_DIR}/configs/gfx942_cdna3.json"
  "${run_wrapper_prefix[@]}" \
    "${rocjitsu_launcher}" --config "${preflight_config}" -- \
    "${child_command_prefix[@]}" true
fi

run_pytest() {
  local timeout_seconds="$1"
  local overall_timeout_seconds="$2"
  local target_name="$3"
  local config_path="$4"
  local skip_config_path="$5"
  local target_artifact_dir="$6"
  local target_cache_dir="$7"
  shift 7
  # The run-wrapper timeout owns the per-test deadline and preserves the
  # command's captured diagnostics. Foreground mode keeps timeout and its child
  # in the supervisor-owned process group. Pytest gets cleanup headroom as a failsafe.
  local pytest_timeout_seconds=$((timeout_seconds + 15))
  local run_wrapper=(
    "${run_wrapper_prefix[@]}"
    setpriv --pdeathsig TERM
    "${corpus_process_supervisor}"
    timeout --foreground --signal=TERM --kill-after=5s "${timeout_seconds}s"
    "${rocjitsu_launcher}"
    --config "${config_path}"
    --
    "${child_command_prefix[@]}"
  )
  local run_wrapper_command
  printf -v run_wrapper_command '%q ' "${run_wrapper[@]}"

  local pytest_cmd=(
    pytest tests/test_corpus.py
    --target "${target_name}"
    --suite "iree,kernels,cts"
    --run-wrapper "${run_wrapper_command% }"
    --skip-tests-config "${skip_config_path}"
    --artifact-directory "${target_artifact_dir}"
    --durations=0
    -vv
    -o "cache_dir=${target_cache_dir}"
    --tb=short
    -n "${worker_count}"
    -o "timeout_func_only=true"
    -o "junit_duration_report=call"
  )
  if (( overall_timeout_seconds > 0 )); then
    timeout --signal=TERM --kill-after=10s "${overall_timeout_seconds}s" \
      "${pytest_cmd[@]}" --timeout "${pytest_timeout_seconds}" "$@"
    return
  fi
  "${pytest_cmd[@]}" --timeout "${pytest_timeout_seconds}" "$@"
}

for target in "${targets[@]}"; do
  read -r name rocjitsu_config skip_tests_config <<< "${target}"
  echo "::group::(${name}) pytest"

  rocjitsu_config_path="${ROCJITSU_SOURCE_DIR}/configs/${rocjitsu_config}"
  skip_tests_config_path="${ROCJITSU_SOURCE_DIR}/tests/corpus/${skip_tests_config}"
  artifact_dir="${corpus_work_dir}/.pytest-artifacts/${name}"
  cache_dir="${corpus_work_dir}/.pytest-cache/${name}"
  junit_xml="${junit_dir}/${name}.xml"

  # Only the soft-timeout run is configured to output a JUnit XML report.
  first_run_status=0
  run_pytest "${soft_timeout_seconds}" 0 "${name}" "${rocjitsu_config_path}" \
    "${skip_tests_config_path}" "${artifact_dir}" "${cache_dir}" \
    --junitxml "${junit_xml}" || first_run_status=$?
  if [[ -f "${junit_xml}" ]]; then
    junit_xml_paths+=("${junit_xml}")
  fi

  if (( first_run_status == 0 )); then
    echo "::endgroup::"
    echo "All (${name}) tests passed."
    continue
  fi

  corpus_test_status=1
  echo "::endgroup::"
  echo "::error::Some (${name}) tests failed."
  echo "::group::(${name}) pytest last-failed summary"
  pytest -o "cache_dir=${cache_dir}" --cache-show="cache/lastfailed" || true
  echo "::endgroup::"

  if [[ "${rerun_failed}" == false ]]; then
    continue
  fi

  # Retry success does not turn CI green.
  echo "::group::(${name}) pytest rerun failed tests"
  if run_pytest "${hard_timeout_seconds}" "${rerun_timeout_seconds}" "${name}" \
       "${rocjitsu_config_path}" "${skip_tests_config_path}" "${artifact_dir}" \
       "${cache_dir}" --last-failed --last-failed-no-failures=none; then
    echo "::endgroup::"
    echo "::warning::Retried (${name}) tests passed."
    continue
  fi
  echo "::endgroup::"
done

# The reporting script's return code does not affect the corpus test status.
if [[ "${warn_perf}" == true ]]; then
  echo "::group::Check for test cases close to the timeout"
  if (( ${#junit_xml_paths[@]} == 0 )); then
    echo "No JUnit report was written; near-timeout reporting has no input."
  else
    report_status=0
    python3 "${ROCJITSU_SOURCE_DIR}/tests/corpus/report-near-timeout-tests.py" \
      --timeout "${soft_timeout_seconds}" \
      "${junit_xml_paths[@]}" || report_status=$?
    case "${report_status}" in
      0)
        ;;
      3)
        echo "::warning::Near-timeout tests found."
        ;;
      *)
        echo "::warning::Near-timeout reporting failed."
        ;;
    esac
  fi
  echo "::endgroup::"
fi

exit "${corpus_test_status}"
