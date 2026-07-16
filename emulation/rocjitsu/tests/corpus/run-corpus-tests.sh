#!/usr/bin/env bash
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

  if "${pytest_cmd[@]}" -n "${worker_count}" -o "timeout_func_only=true" --timeout "${soft_timeout_seconds}"; then
    status_message="All (${name}) tests passed."
  elif [[ "${rerun_failed}" == true ]] && "${pytest_cmd[@]}" --last-failed --timeout "${hard_timeout_seconds}"; then
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
