#!/usr/bin/env bash
set -euo pipefail

: "${ROCM_PATH:?ROCM_PATH must be set}"
: "${ROCJITSU_SOURCE_DIR:?ROCJITSU_SOURCE_DIR must be set}"

WORKER_COUNT="${WORKER_COUNT:-8}"
SOFT_TIMEOUT_SECONDS="${SOFT_TIMEOUT_SECONDS:-30}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-60}"

default_targets=(
  "gfx942 gfx942_cdna3.json gfx942_skip_tests.json"
  "gfx950 gfx950_cdna4.json gfx950_skip_tests.json"
  "gfx1100 gfx1100_w7900.json gfx1100_skip_tests.json"
  "gfx1201 gfx1201_r9700.json gfx1201_skip_tests.json"
  "gfx1250 gfx1250.json gfx1250_skip_tests.json"
)

targets=("$@")
(( ${#targets[@]} )) || targets=("${default_targets[@]}")

corpus_test_status=0
for target in "${targets[@]}"; do
  read -r name rocjitsu_config skip_tests_config <<< "${target}"
  echo "::group::pytest (${name})"

  rocjitsu_config_path="${ROCJITSU_SOURCE_DIR}/configs/${rocjitsu_config}"
  skip_tests_config_path="${ROCJITSU_SOURCE_DIR}/tests/corpus/${skip_tests_config}"
  artifact_dir=".pytest-artifacts/${name}"
  cache_dir=".pytest-cache/${name}"
  target_failed=0

  pytest_cmd=(
    rocjitsu --config "${rocjitsu_config_path}" -- pytest tests/test_corpus.py
    --target "${name}"
    --suite iree,kernels,cts
    --skip-tests-config "${skip_tests_config_path}"
    --artifact-directory "${artifact_dir}"
    --durations=0
    -vv
    -o "cache_dir=${cache_dir}"
    --tb=no
  )

  if "${pytest_cmd[@]}" -n "${WORKER_COUNT}" -o "timeout_func_only=true" --timeout "${SOFT_TIMEOUT_SECONDS}"; then
    status_message="All (${name}) tests passed."
  elif "${pytest_cmd[@]}" --last-failed --timeout "${HARD_TIMEOUT_SECONDS}"; then
    status_message="Retried (${name}) tests passed."
  else
    corpus_test_status=1
    status_message="::warning::Some (${name}) tests failed."
    target_failed=1
  fi
  echo "::endgroup::"
  echo "${status_message}"
  if (( target_failed )); then
    echo "::group::pytest last-failed summary (${name})"
    pytest -o "cache_dir=${cache_dir}" --cache-show="cache/lastfailed" || true
    echo "::endgroup::"
  fi
done

exit "${corpus_test_status}"
