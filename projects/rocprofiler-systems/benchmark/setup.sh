#!/usr/bin/env bash
# Sets up benchmark-suite: checks prerequisites, creates .venv, installs the
# `bench*` CLIs (pip install -e .), and builds the workloads into ./build.
# Scenario JSONs reference workloads as ../../../build/<name>, so the build dir
# must stay at <repo>/build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"
BUILD="${ROOT}/build"
MIN_PY="3.9"
MIN_CMAKE="3.16"

build_type="Release"
gpu="ON"
tests="ON"
clean=0
check_only=0

usage() {
    cat <<EOF
Usage: ./setup.sh [options]
  --check          only check prerequisites, change nothing
  --clean          remove .venv and build/ first
  --no-gpu         skip HIP workloads (BENCH_BUILD_GPU=OFF)
  --no-tests       skip gtest tests (BENCH_BUILD_TESTS=OFF)
  --build-type T   CMake build type (default: Release)
  -h, --help       show this help
Extra CMake args: set CMAKE_ARGS="-DFOO=bar" in the environment.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) check_only=1 ;;
        --clean) clean=1 ;;
        --no-gpu) gpu="OFF" ;;
        --no-tests) tests="OFF" ;;
        --build-type) build_type="${2:?--build-type needs a value}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

errors=0
ok()   { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
fail() { echo "  [FAIL] $*" >&2; errors=$((errors + 1)); }

# true if version $1 >= $2 (dotted numeric)
version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]; }

echo "Checking prerequisites"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for cand in python3 python; do
        if command -v "${cand}" >/dev/null 2>&1; then PYTHON="${cand}"; break; fi
    done
fi
if [[ -z "${PYTHON}" ]]; then
    fail "python3 not found (need >= ${MIN_PY})"
else
    py_ver="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if version_ge "${py_ver}" "${MIN_PY}"; then ok "python ${py_ver} (${PYTHON})"
    else fail "python ${py_ver} is too old (need >= ${MIN_PY})"; fi
    if "${PYTHON}" -c 'import venv, ensurepip' 2>/dev/null; then ok "python venv/ensurepip"
    else fail "python venv module missing (Debian/Ubuntu: apt install python3-venv)"; fi
fi

if command -v cmake >/dev/null 2>&1; then
    cm_ver="$(cmake --version | head -n1 | awk '{print $3}')"
    if version_ge "${cm_ver}" "${MIN_CMAKE}"; then ok "cmake ${cm_ver}"
    else fail "cmake ${cm_ver} is too old (need >= ${MIN_CMAKE})"; fi
else
    fail "cmake not found (need >= ${MIN_CMAKE})"
fi

if command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; then
    ok "C++ compiler"
else
    fail "no C++ compiler (c++/g++/clang++) found"
fi

if command -v ninja >/dev/null 2>&1; then ok "ninja"
elif command -v make >/dev/null 2>&1; then ok "make"
else fail "neither ninja nor make found"; fi

if [[ "${gpu}" == "ON" ]]; then
    if command -v amdclang++ >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1; then
        ok "HIP compiler ($(command -v amdclang++ || command -v hipcc))"
    else
        warn "no amdclang++/hipcc in PATH: GPU workloads will be skipped (add ROCm to PATH or use --no-gpu)"
    fi
fi

if command -v rocprof-sys-run >/dev/null 2>&1; then ok "rocprof-sys-run ($(command -v rocprof-sys-run))"
else warn "rocprof-sys-run not in PATH: only needed for the profiled scenarios"; fi

if [[ "${errors}" -gt 0 ]]; then
    echo "${errors} required prerequisite(s) missing; aborting." >&2
    exit 1
fi
if [[ "${check_only}" -eq 1 ]]; then
    echo "Prerequisites OK."
    exit 0
fi

if [[ "${clean}" -eq 1 ]]; then
    echo "Cleaning ${VENV} and ${BUILD}"
    rm -rf "${VENV}" "${BUILD}"
fi

echo "Python environment: ${VENV}"
[[ -x "${VENV}/bin/python" ]] || "${PYTHON}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --quiet --upgrade pip
"${VENV}/bin/python" -m pip install --quiet -e "${ROOT}[dev]"

echo "Workloads: ${BUILD} (${build_type}, GPU=${gpu}, tests=${tests})"
# shellcheck disable=SC2086  # CMAKE_ARGS is intentionally word-split
cmake -S "${ROOT}/workloads" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DBENCH_BUILD_GPU="${gpu}" \
    -DBENCH_BUILD_TESTS="${tests}" \
    ${CMAKE_ARGS:-}
cmake --build "${BUILD}" --parallel

echo
echo "Built workloads:"
find "${BUILD}" -maxdepth 1 -type f -perm -u+x -printf '  %f\n' | sort
cat <<EOF

Done. Activate with:
  source ${VENV}/bin/activate
Then try:
  bench-snapshot --help
EOF
