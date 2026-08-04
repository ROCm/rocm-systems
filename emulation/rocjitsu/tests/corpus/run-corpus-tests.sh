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
#   --sanitizer MODE     Launcher instrumentation: none, clang-asan, or gcc-asan
#   --rerun-failed       Rerun only tests that failed the first pass
#
# Environment variables:
#   ROCM_PATH            Required ROCm installation root
#   ROCJITSU_SOURCE_DIR  Required rocjitsu source directory

set -euo pipefail

: "${ROCM_PATH:?ROCM_PATH must be set}"
: "${ROCJITSU_SOURCE_DIR:?ROCJITSU_SOURCE_DIR must be set}"

worker_count=8
soft_timeout_seconds=30
hard_timeout_seconds=60
rerun_failed=false
sanitizer_mode=none

usage() {
  echo "Usage: $0 [--workers N] [--soft-timeout N] [--hard-timeout N]" \
    "[--sanitizer none|clang-asan|gcc-asan] [--rerun-failed]" >&2
}

targets=(
  "gfx942 gfx942_cdna3.json gfx942_skip_tests.json"
  "gfx950 gfx950_cdna4.json gfx950_skip_tests.json"
  "gfx1100 gfx1100_w7900.json gfx1100_skip_tests.json"
  "gfx1201 gfx1201_r9700.json gfx1201_skip_tests.json"
  "gfx1250 gfx1250.json gfx1250_skip_tests.json"
)

while (( $# )); do
  case "$1" in
    --workers)
      worker_count="$2"
      shift 2
      ;;
    --soft-timeout)
      soft_timeout_seconds="$2"
      shift 2
      ;;
    --hard-timeout)
      hard_timeout_seconds="$2"
      shift 2
      ;;
    --sanitizer)
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
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

corpus_test_status=0
corpus_work_dir="$(pwd -P)"
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
  # LeakSanitizer's stop-the-world teardown scan stalls on multi-gigabyte HIP
  # process mappings. Keep ASan and UBSan enabled while omitting that final scan.
  corpus_asan_options="${ASAN_OPTIONS:+${ASAN_OPTIONS}:}detect_leaks=0"
  run_wrapper_prefix+=(
    "ASAN_OPTIONS=${corpus_asan_options}"
    "ASAN_SYMBOLIZER_PATH=${asan_symbolizer}"
  )
fi

# Clang ASan launches need HIP loaded when the child process starts.
# Keep the preload in the launched subtree so the launcher does not initialize
# HIP itself.
launcher_preload_args=()
if [[ "${sanitizer_mode}" == clang-asan ]]; then
  hip_runtime="${ROCM_PATH}/lib/libamdhip64.so"
  if [[ ! -f "${hip_runtime}" ]]; then
    echo "Could not resolve HIP runtime for Clang ASan corpus preload" >&2
    exit 1
  fi
  launcher_preload_args=(--preload "${hip_runtime}")
fi

# Full corpus coverage remains in the release lane. Instrumenting the
# interpreter makes the larger generated workloads too slow for a bounded CI
# job, so sanitizer lanes run a stable runtime smoke set on every target.
sanitizer_pytest_args=()
if [[ "${sanitizer_mode}" != none ]]; then
  sanitizer_pytest_args=(
    -k
    "fpsan_build_canary or fpsan_cast_test or fpsan_core_test"
  )
fi

run_pytest() {
  local timeout_seconds="$1"
  shift
  local run_wrapper=(
    "${run_wrapper_prefix[@]}"
    setpriv --pdeathsig TERM
    "${corpus_process_supervisor}"
    timeout --signal=TERM --kill-after=5s "${timeout_seconds}s"
    "${rocjitsu_launcher}"
    --config "${rocjitsu_config_path}"
    "${launcher_preload_args[@]}"
    --
  )
  local run_wrapper_command
  printf -v run_wrapper_command '%q ' "${run_wrapper[@]}"

  local pytest_cmd=(
    pytest tests/test_corpus.py
    --target "${name}"
    --suite iree,kernels,cts
    --run-wrapper "${run_wrapper_command% }"
    --skip-tests-config "${skip_tests_config_path}"
    --artifact-directory "${artifact_dir}"
    --durations=0
    -vv
    -o "cache_dir=${cache_dir}"
    --tb=short
    -n "${worker_count}"
    -o "timeout_func_only=true"
  )
  "${pytest_cmd[@]}" --timeout "${timeout_seconds}" "$@"
}

for target in "${targets[@]}"; do
  read -r name rocjitsu_config skip_tests_config <<< "${target}"
  echo "::group::(${name}) pytest"

  rocjitsu_config_path="${ROCJITSU_SOURCE_DIR}/configs/${rocjitsu_config}"
  skip_tests_config_path="${ROCJITSU_SOURCE_DIR}/tests/corpus/${skip_tests_config}"
  artifact_dir="${corpus_work_dir}/.pytest-artifacts/${name}"
  cache_dir="${corpus_work_dir}/.pytest-cache/${name}"
  if run_pytest "${soft_timeout_seconds}"; then
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
  if run_pytest "${hard_timeout_seconds}" --last-failed --last-failed-no-failures=none; then
    echo "::endgroup::"
    echo "::warning::Retried (${name}) tests passed."
    continue
  fi
  echo "::endgroup::"
done

exit "${corpus_test_status}"
