#!/usr/bin/env bash
#
# Validation driver for the queue-callback-removal branches.
#
# Each of the four services (counter collection, SPM, thread trace, PC sampling) has been
# moved off the per-queue callback registry and onto explicit hooks called from the HSA
# write interceptor and the async signal handler. The interesting failure modes are all
# timing dependent -- a dispatch that completes after its context stops, two contexts
# scoped to different agents draining at once, serialization references that outlive the
# context that took them -- so they only show up on a machine with real GPU agents.
#
# This script runs, per branch: the formatting gate that CI runs, a build, the unit tests
# for the four services, the counter/ATT/PC-sampling integration tests, and optionally the
# benchmark suite to compare dispatch overhead against the base branch.
#
# Usage:
#   tools/callback-removal/validate.sh [--perf] [--base <ref>] [branch ...]
#
# With no branch arguments it validates the four review-fix branches. Results go to
# tools/callback-removal/results/<branch>/ and a summary is printed at the end.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDK_DIR="${REPO_ROOT}/projects/rocprofiler-sdk"
RESULTS_ROOT="${REPO_ROOT}/tools/callback-removal/results"

BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CTEST_JOBS="${CTEST_JOBS:-4}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
RUN_PERF=0
BASE_REF="develop"

DEFAULT_BRANCHES=(
    cursor/pr8891-counters-review-fixes-5c04
    cursor/pr8887-spm-review-fixes-5c04
    cursor/pr8790-att-review-fixes-5c04
    cursor/pr8895-pcs-review-fixes-5c04
)

# Unit test suites that cover the migrated hook paths.
UNIT_TEST_REGEX='counters|spm|thread.trace|thread_trace|pcs|pc_sampling|queue'
# Integration tests that exercise a real dispatch through the interceptor.
INTEGRATION_TEST_REGEX='counter-collection|counter_collection|thread-trace|att|pc-sampling|pc_sampling'

branches=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --perf) RUN_PERF=1; shift ;;
        --base) BASE_REF="$2"; shift 2 ;;
        -h|--help) awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) branches+=("$1"); shift ;;
    esac
done
[[ ${#branches[@]} -eq 0 ]] && branches=("${DEFAULT_BRANCHES[@]}")

declare -A RESULT_FORMAT RESULT_BUILD RESULT_UNIT RESULT_INTEGRATION

log() { printf '\n==> %s\n' "$*"; }

require() {
    command -v "$1" >/dev/null 2>&1 || { echo "required tool not found: $1" >&2; exit 1; }
}

# The formatting job pins clang-format 11; any other version reformats the whole tree and
# produces a diff that looks like the branch's fault.
check_format_tool() {
    require clang-format
    local version
    version="$(clang-format --version | grep -oE '[0-9]+' | head -1)"
    if [[ "${version}" != "11" ]]; then
        echo "clang-format ${version} found, but CI pins version 11." >&2
        echo "install with: python3 -m pip install 'clang-format==11.1.0'" >&2
        exit 1
    fi
}

run_format_gate() {
    local out="$1"
    ( cd "${SDK_DIR}" || exit 1
      mapfile -t files < <(find samples source tests benchmark -type f 2>/dev/null \
          | grep -E '\.(h|hpp|hh|c|cc|cpp)(|\.in)$')
      clang-format -i "${files[@]}" )
    if [[ -n "$(cd "${REPO_ROOT}" && git diff --name-only)" ]]; then
        (cd "${REPO_ROOT}" && git diff) > "${out}"
        (cd "${REPO_ROOT}" && git diff --name-only)
        (cd "${REPO_ROOT}" && git checkout -- .)
        return 1
    fi
    return 0
}

detect_gpu_target() {
    if command -v amdgpu-arch >/dev/null 2>&1; then
        amdgpu-arch | head -1
    elif command -v "${ROCM_PATH}/llvm/bin/amdgpu-arch" >/dev/null 2>&1; then
        "${ROCM_PATH}/llvm/bin/amdgpu-arch" | head -1
    elif command -v rocminfo >/dev/null 2>&1; then
        rocminfo | grep -oE 'gfx[0-9a-f]+' | head -1
    fi
}

build_branch() {
    local build_dir="$1" log_file="$2" gpu_target="$3"
    local -a gpu_arg=()
    [[ -n "${gpu_target}" ]] && gpu_arg=(-DGPU_TARGETS="${gpu_target}")

    cmake -B "${build_dir}" -S "${SDK_DIR}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DROCPROFILER_BUILD_TESTS=ON \
        -DROCPROFILER_BUILD_SAMPLES=ON \
        -DROCPROFILER_BUILD_DOCS=OFF \
        -DROCPROFILER_DEP_ROCMCORE=ON \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH};${ROCM_PATH}/llvm" \
        -DPython3_EXECUTABLE="$(command -v python3)" \
        "${gpu_arg[@]}" >>"${log_file}" 2>&1 || return 1

    cmake --build "${build_dir}" --target all --parallel "${BUILD_JOBS}" >>"${log_file}" 2>&1
}

run_ctest() {
    local build_dir="$1" regex="$2" log_file="$3" extra="${4:-}"
    # shellcheck disable=SC2086
    ctest --test-dir "${build_dir}" \
        --parallel "${CTEST_JOBS}" \
        --output-on-failure \
        -R "${regex}" ${extra} >>"${log_file}" 2>&1
}

run_perf() {
    local build_dir="$1" out_dir="$2"
    log "benchmark suite (dispatch overhead)"
    cmake -B "${build_dir}" -S "${SDK_DIR}/benchmark" >>"${out_dir}/perf.log" 2>&1 || return 1
    cmake --build "${build_dir}" --parallel "${BUILD_JOBS}" >>"${out_dir}/perf.log" 2>&1 || return 1
    ( cd "${build_dir}" && PATH="${PWD}/bin:${PATH}" \
        rocprofv3-benchmark -i "${SDK_DIR}/benchmark/example.yml" -n 5 ) >>"${out_dir}/perf.log" 2>&1 || return 1
    if [[ -f "${build_dir}/benchmark.db" ]] && command -v sqlite3 >/dev/null 2>&1; then
        sqlite3 "${build_dir}/benchmark.db" \
            'SELECT * FROM benchmark_statistics;' > "${out_dir}/benchmark_statistics.txt"
        cp "${build_dir}/benchmark.db" "${out_dir}/benchmark.db"
    fi
}

require git
require cmake
require ctest
check_format_tool

cd "${REPO_ROOT}" || exit 1
if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree is dirty; commit or stash before validating" >&2
    exit 1
fi
ORIGINAL_REF="$(git rev-parse --abbrev-ref HEAD)"
restore() { cd "${REPO_ROOT}" && git checkout -q "${ORIGINAL_REF}" 2>/dev/null; }
trap restore EXIT

GPU_TARGET="$(detect_gpu_target)"
log "GPU target: ${GPU_TARGET:-<not detected, letting cmake decide>}"
[[ -e /dev/kfd ]] || log "WARNING: /dev/kfd missing; GPU tests will skip rather than validate"

for branch in "${branches[@]}"; do
    log "validating ${branch}"
    out_dir="${RESULTS_ROOT}/${branch//\//_}"
    mkdir -p "${out_dir}"
    build_dir="${REPO_ROOT}/build-${branch//\//_}"

    if ! git checkout -q "${branch}" 2>>"${out_dir}/git.log"; then
        echo "  cannot check out ${branch}; skipping" >&2
        RESULT_FORMAT[$branch]="n/a"; RESULT_BUILD[$branch]="checkout failed"
        RESULT_UNIT[$branch]="n/a"; RESULT_INTEGRATION[$branch]="n/a"
        continue
    fi

    log "formatting gate"
    if run_format_gate "${out_dir}/format.diff"; then
        RESULT_FORMAT[$branch]="pass"
    else
        RESULT_FORMAT[$branch]="FAIL (see format.diff)"
    fi

    log "build"
    : > "${out_dir}/build.log"
    if build_branch "${build_dir}" "${out_dir}/build.log" "${GPU_TARGET}"; then
        RESULT_BUILD[$branch]="pass"
    else
        RESULT_BUILD[$branch]="FAIL (see build.log)"
        RESULT_UNIT[$branch]="skipped"; RESULT_INTEGRATION[$branch]="skipped"
        continue
    fi

    log "unit tests"
    : > "${out_dir}/unit.log"
    if run_ctest "${build_dir}" "${UNIT_TEST_REGEX}" "${out_dir}/unit.log" '-L unittests'; then
        RESULT_UNIT[$branch]="pass"
    else
        RESULT_UNIT[$branch]="FAIL (see unit.log)"
    fi
    # A skip is not a pass: these are exactly the tests that need real agents.
    grep -c 'Skipped' "${out_dir}/unit.log" > "${out_dir}/unit-skipped.count" 2>/dev/null

    log "integration tests"
    : > "${out_dir}/integration.log"
    if run_ctest "${build_dir}" "${INTEGRATION_TEST_REGEX}" "${out_dir}/integration.log"; then
        RESULT_INTEGRATION[$branch]="pass"
    else
        RESULT_INTEGRATION[$branch]="FAIL (see integration.log)"
    fi

    if [[ ${RUN_PERF} -eq 1 ]]; then
        run_perf "${REPO_ROOT}/build-benchmark-${branch//\//_}" "${out_dir}" \
            || echo "  benchmark run failed; see perf.log" >&2
    fi
done

if [[ ${RUN_PERF} -eq 1 ]]; then
    log "baseline benchmark on ${BASE_REF} for comparison"
    if git checkout -q "${BASE_REF}" 2>/dev/null; then
        mkdir -p "${RESULTS_ROOT}/${BASE_REF//\//_}"
        run_perf "${REPO_ROOT}/build-benchmark-base" "${RESULTS_ROOT}/${BASE_REF//\//_}" \
            || echo "  baseline benchmark failed" >&2
    fi
fi

printf '\n%s\n' "================ summary ================"
printf '%-46s %-10s %-24s %-24s %s\n' "branch" "format" "build" "unit" "integration"
for branch in "${branches[@]}"; do
    printf '%-46s %-10s %-24s %-24s %s\n' \
        "${branch}" \
        "${RESULT_FORMAT[$branch]:-?}" \
        "${RESULT_BUILD[$branch]:-?}" \
        "${RESULT_UNIT[$branch]:-?}" \
        "${RESULT_INTEGRATION[$branch]:-?}"
done
printf '\nlogs: %s\n' "${RESULTS_ROOT}"

overall=0
for branch in "${branches[@]}"; do
    for result in "${RESULT_FORMAT[$branch]:-}" "${RESULT_BUILD[$branch]:-}" \
                  "${RESULT_UNIT[$branch]:-}" "${RESULT_INTEGRATION[$branch]:-}"; do
        [[ "${result}" == FAIL* ]] && overall=1
    done
done
exit ${overall}
