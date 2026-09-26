#!/bin/bash
# =============================================================================
# Multi-ROCm-version rocSHMEM build + smoketest driver
#
# Iterates over all /opt/rocm-* installations, builds rocshmem against each,
# and optionally runs a ping-pong functional test + Python wheel smoke test.
#
# Usage:
#   build_and_test.sh                    # default versions, full test (needs GPU)
#   build_and_test.sh --build-only       # default versions, compile + wheel only
#   build_and_test.sh --no-wheel         # skip Python wheel build + test
#   build_and_test.sh --all              # all installed versions
#   build_and_test.sh 7.13               # single version
#   build_and_test.sh --build-only 6.3.3 7.2.4  # specific versions, compile only
# =============================================================================
set -euo pipefail

# Default versions to build/test (subset of installed)
DEFAULT_VERSIONS="6.3.3 7.0.2 7.2.4 7.13 7.14"

BUILD_ONLY=false
ALL_VERSIONS=false
NO_WHEEL=false
ROCM_VERSIONS=()

for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=true ;;
        --all) ALL_VERSIONS=true ;;
        --no-wheel) NO_WHEEL=true ;;
        *) ROCM_VERSIONS+=("$arg") ;;
    esac
done

# Resolve a version string to a directory path
resolve_rocm_dir() {
    local ver="$1"
    for candidate in "/opt/rocm-${ver}" "/opt/rocm/core-${ver}"; do
        if [[ -d "$candidate/bin" ]]; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

if [[ "$ALL_VERSIONS" == true ]]; then
    ROCM_VERSIONS=()
    for d in /opt/rocm-*; do
        [[ -d "$d/bin" ]] && ROCM_VERSIONS+=("$d")
    done
    for d in /opt/rocm/core-*.*; do
        [[ -d "$d/bin" ]] && ROCM_VERSIONS+=("$d")
    done
elif [[ ${#ROCM_VERSIONS[@]} -eq 0 ]]; then
    # Use defaults
    for ver in $DEFAULT_VERSIONS; do
        ROCM_VERSIONS+=("$ver")
    done
fi

# Resolve version strings to paths
RESOLVED=()
for entry in "${ROCM_VERSIONS[@]}"; do
    if [[ "$entry" == /opt/* ]]; then
        RESOLVED+=("$entry")
    else
        dir=$(resolve_rocm_dir "$entry")
        if [[ -n "$dir" ]]; then
            RESOLVED+=("$dir")
        else
            echo "WARNING: ROCm $entry not found, skipping"
        fi
    fi
done
ROCM_VERSIONS=("${RESOLVED[@]}")

SRC="${ROCSHMEM_SRC:-/app/rocshmem}"
# The rocshmem4py Python project is a standalone project that now lives outside
# the rocSHMEM C++ tree (rocm-systems/python/rocshmem). Default to its location
# relative to $SRC; the wheel step below is skipped when this path is absent
# (e.g. when only the rocSHMEM project is mounted).
PYSRC="${ROCSHMEM_PY_SRC:-$SRC/../../python/rocshmem}"
OMPI="${OMPI_INSTALL:-/opt/ompi}"

HAS_GPU=false
[[ -e /dev/kfd ]] && HAS_GPU=true

PASS=()
FAIL=()

run_step() {
    local version="$1" step="$2"
    shift 2
    echo "--- [$version] $step ---"
    if "$@"; then
        return 0
    else
        echo "FAILED: [$version] $step"
        return 1
    fi
}

for rocm_dir in "${ROCM_VERSIONS[@]}"; do
    [[ -d "$rocm_dir" ]] || continue
    version=$(basename "$rocm_dir" | sed 's/^rocm-//;s/^core-//')
    echo ""
    echo "========================================"
    echo "  ROCm $version ($rocm_dir)"
    echo "========================================"

    builddir="/app/build-${version}"
    installdir="/app/install-${version}"
    ok=true

    # -----------------------------------------------------------------
    # 1. CMake configure + build + install
    # Prepend rocm_dir to PATH and LD_LIBRARY_PATH so compiler wrappers,
    # hipconfig, and shared libraries resolve to the target ROCm version
    # rather than whatever /opt/rocm symlinks to.
    # -----------------------------------------------------------------
    mkdir -p "$builddir"
    if ! run_step "$version" "cmake configure" \
        env PATH="${rocm_dir}/bin:${rocm_dir}/llvm/bin:${PATH}" \
            LD_LIBRARY_PATH="${rocm_dir}/lib:${LD_LIBRARY_PATH:-}" \
        cmake -S "$SRC" -B "$builddir" \
            -DCMAKE_PREFIX_PATH="$rocm_dir" \
            -DCMAKE_INSTALL_PREFIX="$installdir" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DUSE_RO=ON -DUSE_IPC=ON \
            -DUSE_GDA=ON -DGDA_MLX5=ON -DGDA_BNXT=ON -DGDA_IONIC=ON \
            -DUSE_EXTERNAL_MPI=OFF -DMPI_ROOT="$OMPI" \
            -DBUILD_FUNCTIONAL_TESTS=ON; then
        FAIL+=("$version:configure")
        continue
    fi

    if ! run_step "$version" "cmake build" \
        env PATH="${rocm_dir}/bin:${rocm_dir}/llvm/bin:${PATH}" \
            LD_LIBRARY_PATH="${rocm_dir}/lib:${LD_LIBRARY_PATH:-}" \
        cmake --build "$builddir" --parallel "$(nproc)"; then
        FAIL+=("$version:build")
        continue
    fi

    if ! run_step "$version" "cmake install" \
        cmake --install "$builddir"; then
        FAIL+=("$version:install")
        continue
    fi

    # Show NUMA detection result
    echo "--- [$version] NUMA detection ---"
    grep "NUMA_DIR\|_NUMA_LIB\|_NUMA_INC" "$builddir/CMakeCache.txt" 2>/dev/null \
        | grep -v "ADVANCED\|^#" || true

    # -----------------------------------------------------------------
    # 2. Examples build (external package smoke test)
    # Builds the examples directory as a standalone CMake project against
    # the installed rocshmem, validating the exported cmake config.
    # -----------------------------------------------------------------
    examplesdir="/app/build-examples-${version}"
    if ! run_step "$version" "examples build" \
        env PATH="${rocm_dir}/bin:${rocm_dir}/llvm/bin:${PATH}" \
            LD_LIBRARY_PATH="${rocm_dir}/lib:${LD_LIBRARY_PATH:-}" \
        cmake -S "$SRC/examples" -B "$examplesdir" \
            -Drocshmem_ROOT="${installdir}" \
            -DCMAKE_PREFIX_PATH="${installdir};${rocm_dir}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DMPI_ROOT="$OMPI"; then
        FAIL+=("$version:examples-configure")
        ok=false
    elif ! run_step "$version" "examples compile" \
        cmake --build "$examplesdir" --parallel "$(nproc)"; then
        FAIL+=("$version:examples-compile")
        ok=false
    fi

    # -----------------------------------------------------------------
    # 3. Functional smoke test (ping-pong, 2 PEs)
    # -----------------------------------------------------------------
    if [[ "$BUILD_ONLY" == false && "$HAS_GPU" == true ]]; then
        run_step "$version" "putnbi test" \
            env LD_LIBRARY_PATH="${rocm_dir}/lib:${LD_LIBRARY_PATH}" \
            mpiexec -n 2 "$builddir/tests/functional_tests/rocshmem_functional_tests" \
                -a 31 -w 1 -z 256 \
            || { FAIL+=("$version:putnbi"); ok=false; }
    fi

    # -----------------------------------------------------------------
    # 3. Python wheel build
    # -----------------------------------------------------------------
    if [[ "$NO_WHEEL" == false && -d "$PYSRC" ]]; then
        run_step "$version" "python wheel" \
            env ROCM_PATH="$rocm_dir" CMAKE_PREFIX_PATH="$installdir" \
                pip install --no-build-isolation \
                    -e "$PYSRC" \
            || { FAIL+=("$version:wheel"); ok=false; }

        # Python smoke test (needs GPU, run under mpiexec so OMPI_COMM_WORLD_SIZE
        # is set and rocshmem initializes correctly with 2 PEs)
        if [[ "$BUILD_ONLY" == false && "$HAS_GPU" == true ]]; then
            run_step "$version" "python smoke test" \
                env ROCM_PATH="$rocm_dir" CMAKE_PREFIX_PATH="$installdir" \
                    LD_LIBRARY_PATH="${rocm_dir}/lib:${LD_LIBRARY_PATH:-}" \
                mpiexec -n 2 python3 -m pytest "$PYSRC/tests/test_smoke.py" -v \
                || { FAIL+=("$version:pytest"); ok=false; }
        fi
    fi

    $ok && PASS+=("$version")
done

# -----------------------------------------------------------------
# Summary
# -----------------------------------------------------------------
echo ""
echo "========================================"
echo "  SUMMARY"
echo "========================================"
[[ ${#PASS[@]} -gt 0 ]] && echo "PASS: ${PASS[*]}"
if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo "FAIL: ${FAIL[*]}"
    exit 1
fi
echo "All ROCm versions passed."
