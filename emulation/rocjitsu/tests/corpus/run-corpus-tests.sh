#!/usr/bin/env bash
set -euo pipefail

: "${ROCM_PATH:?ROCM_PATH must be set}"
: "${ROCJITSU_SOURCE_DIR:?ROCJITSU_SOURCE_DIR must be set}"
: "${ROCJITSU_CORPUS_TEST_DIR:?ROCJITSU_CORPUS_TEST_DIR must be set}"

if [[ "$#" -eq 0 ]]; then
  echo "::error::No corpus targets provided."
  exit 1
fi

targets=("$@")
corpus_status=0
corpus_results=""

finish_target() {
  local target_status="$1"
  if [[ "${target_status}" -ne 0 ]]; then
    corpus_status=1
  fi
  corpus_results+="${name}: $([[ "${target_status}" -eq 0 ]] && echo PASS || echo FAIL)"$'\n'
  echo "::endgroup::"
}

SOFT_TIMEOUT_SECONDS="${SOFT_TIMEOUT_SECONDS:-15}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-30}"
MAX_SOFT_TIMEOUT_RETRIES="${MAX_SOFT_TIMEOUT_RETRIES:-5}"

for target in "${targets[@]}"; do
  read -r name rocjitsu_config skip_tests_config <<< "${target}"
  echo "::group::pytest (${name})"
  rocjitsu_config_path="${ROCJITSU_SOURCE_DIR}/configs/${rocjitsu_config}"
  skip_tests_config_path="${ROCJITSU_CORPUS_TEST_DIR}/${skip_tests_config}"
  artifact_dir=".pytest-artifacts/${name}"
  cache_dir=".pytest-cache/${name}"
  timeout_failures_config="${artifact_dir}/timeout-failures.json"
  non_timeout_failures_config="${artifact_dir}/non-timeout-failures.json"
  common_pytest_args=(
    --target "${name}"
    --suite iree,kernels,cts
    --skip-tests-config "${skip_tests_config_path}"
    --artifact-directory "${artifact_dir}"
    --durations=0
    -o "cache_dir=${cache_dir}"
  )

  if rocjitsu \
    --config "${rocjitsu_config_path}" \
    -- pytest \
    tests/test_corpus.py \
    "${common_pytest_args[@]}" \
    --timeout-failures-config "${timeout_failures_config}" \
    --non-timeout-failures-config "${non_timeout_failures_config}" \
    --timeout "${SOFT_TIMEOUT_SECONDS}"; then
    finish_target 0
    continue
  fi

  timeout_failures_json="$(tr -d '[:space:]' < "${timeout_failures_config}")"
  non_timeout_failures_json="$(tr -d '[:space:]' < "${non_timeout_failures_config}")"
  if [[ "${non_timeout_failures_json}" != "{}" ]]; then
    echo "::error::Corpus tests had regular failures; not retrying this target."
    finish_target 1
    continue
  fi

  if [[ "${timeout_failures_json}" == "{}" ]]; then
    echo "::error::Corpus tests failed, but no pytest-timeout failures were recorded."
    finish_target 1
    continue
  fi

  timeout_failure_count="$(jq '[.[] | length] | add // 0' "${timeout_failures_config}")"
  if [[ "${timeout_failure_count}" -gt "${MAX_SOFT_TIMEOUT_RETRIES}" ]]; then
    echo "::error title=Too many pytest-timeout failures::Recorded ${timeout_failure_count} pytest-timeout failures, which is more than the retry limit of ${MAX_SOFT_TIMEOUT_RETRIES}; not retrying this target."
    finish_target 1
    continue
  fi

  target_status=1
  echo "::warning::Retrying ${timeout_failure_count} pytest-timeout failure(s) with timeout=${HARD_TIMEOUT_SECONDS}s."
  if rocjitsu \
    --config "${rocjitsu_config_path}" \
    -- pytest \
    tests/test_corpus.py -vv \
    "${common_pytest_args[@]}" \
    --run-tests-config "${timeout_failures_config}" \
    --timeout "${HARD_TIMEOUT_SECONDS}"; then
    echo "::warning::Retried pytest-timeout failures passed."
    target_status=0
  else
    echo "::error::Retried tests that previously timed out still failed with timeout=${HARD_TIMEOUT_SECONDS}s. Check the pytest log to see whether they timed out again or failed normally."
  fi
  finish_target "${target_status}"
done

echo "::group::Corpus test status"
printf '%s' "${corpus_results}"
echo "::endgroup::"
exit "${corpus_status}"
