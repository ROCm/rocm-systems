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
rerun_failed=false
warn_perf=false

usage() {
  echo "Usage: $0 [--workers N] [--soft-timeout N] [--hard-timeout N] [--rerun-failed] [--warn-perf]" >&2
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

corpus_test_status=0
corpus_work_dir="$(pwd -P)"
junit_dir="${corpus_work_dir}/.pytest-artifacts/junit"
junit_xml_paths=()

# A report left by an earlier run would otherwise be reported as this run's.
rm -rf "${junit_dir}"

for target in "${targets[@]}"; do
  read -r name rocjitsu_config skip_tests_config <<< "${target}"
  echo "::group::(${name}) pytest"

  rocjitsu_config_path="${ROCJITSU_SOURCE_DIR}/configs/${rocjitsu_config}"
  skip_tests_config_path="${ROCJITSU_SOURCE_DIR}/tests/corpus/${skip_tests_config}"
  artifact_dir="${corpus_work_dir}/.pytest-artifacts/${name}"
  cache_dir="${corpus_work_dir}/.pytest-cache/${name}"
  junit_xml="${junit_dir}/${name}.xml"

  pytest_cmd=(
    rocjitsu --config "${rocjitsu_config_path}" -- pytest tests/test_corpus.py
    --target "${name}"
    --suite iree,kernels,cts
    --skip-tests-config "${skip_tests_config_path}"
    --artifact-directory "${artifact_dir}"
    --durations=0
    -vv
    -o "cache_dir=${cache_dir}"
    --tb=short
    -n "${worker_count}"
    -o "timeout_func_only=true"
    -o "junit_duration_report=call"
  )

  # Only the soft-timeout run is configured to output a JUnit XML report.
  first_run_status=0
  "${pytest_cmd[@]}" --timeout "${soft_timeout_seconds}" \
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
  if "${pytest_cmd[@]}" --last-failed --last-failed-no-failures=none --timeout "${hard_timeout_seconds}"; then
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
