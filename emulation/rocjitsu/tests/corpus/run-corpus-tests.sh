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

usage() {
  echo "Usage: $0 [--workers N] [--soft-timeout N] [--hard-timeout N] [--rerun-failed]" >&2
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
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

corpus_test_status=0
for target in "${targets[@]}"; do
  read -r name rocjitsu_config skip_tests_config <<< "${target}"
  echo "::group::(${name}) pytest"

  rocjitsu_config_path="${ROCJITSU_SOURCE_DIR}/configs/${rocjitsu_config}"
  skip_tests_config_path="${ROCJITSU_SOURCE_DIR}/tests/corpus/${skip_tests_config}"
  artifact_dir=".pytest-artifacts/${name}"
  cache_dir=".pytest-cache/${name}"

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
  )

  if "${pytest_cmd[@]}" --timeout "${soft_timeout_seconds}"; then
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

exit "${corpus_test_status}"
