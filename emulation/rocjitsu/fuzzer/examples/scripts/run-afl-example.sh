#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run-afl-example.sh [target] [afl-fuzz options...]

Targets:
  rocblas-sgemm     Persistent rocBLAS SGEMM harness with device branch coverage.
  rocfft-c2c        Persistent rocFFT C2C harness with device branch coverage.
  rocrand-uniform   Persistent rocRAND uniform harness with device branch coverage.
  rocsparse-spmv    rocSPARSE CSR SpMV harness with device branch coverage.
  rocsolver-getrf   Persistent rocSOLVER GETRF harness with device branch coverage.
  miopen-activation Persistent MIOpen activation harness with device branch coverage.

Examples:
  scripts/run-afl-example.sh rocblas-sgemm -V 30
  scripts/run-afl-example.sh rocfft-c2c -V 30
  scripts/run-afl-example.sh rocsparse-spmv -V 30

Environment overrides:
  ROCFUZZ_AFLPP_ROOT       AFL++ checkout. Defaults to ../../third_party/AFLplusplus.
  ROCFUZZ_AFL_BUILD_DIR    AFL CMake build dir. Defaults to build-afl.
  ROCFUZZ_AFL_OUT_DIR      AFL output dir. Defaults to afl-out/<target>.
  ROCFUZZ_AFL_PRELOAD      rocjitsu AFL DBI preload .so path.
  ROCFUZZ_AFL_REPORT       Optional markdown report path to write after afl-fuzz exits.
  ROCFUZZ_BUILD_JOBS       Parallel build jobs. Defaults to nproc.
  ROCFUZZ_REQUIRE_DEVICE_EDGES
                            Set to 1 for short wiring checks that should fail
                            when no device edge slots are observed.

Run scripts/setup-therock.sh first so therock.env exists.
The wrapper defaults AFL_AUTORESUME, AFL_SKIP_CPUFREQ, and
AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES to 1 for local example runs.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
examples_dir="$(cd -- "${script_dir}/.." && pwd)"
rocjitsu_dir="$(cd -- "${examples_dir}/../.." && pwd)"

target="${1:-rocfft-c2c}"
if [[ "${target}" == "-h" || "${target}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 ]]; then
    shift
fi
afl_args=("$@")

therock_env="${examples_dir}/therock.env"
if [[ ! -f "${therock_env}" ]]; then
    echo "error: missing ${therock_env}; run scripts/setup-therock.sh first" >&2
    exit 1
fi
# shellcheck source=/dev/null
. "${therock_env}"

afl_root="${ROCFUZZ_AFLPP_ROOT:-${rocjitsu_dir}/third_party/AFLplusplus}"
afl_cxx="${ROCFUZZ_AFL_CXX:-${afl_root}/afl-clang-fast++}"
afl_fuzz="${ROCFUZZ_AFL_FUZZ:-${afl_root}/afl-fuzz}"
preload="${ROCFUZZ_AFL_PRELOAD:-${rocjitsu_dir}/build/fuzzer/afl-dbi/librocjitsu_afl_preload.so}"
build_dir="${ROCFUZZ_AFL_BUILD_DIR:-${examples_dir}/build-afl}"
jobs="${ROCFUZZ_BUILD_JOBS:-$(nproc)}"

if [[ ! -x "${afl_cxx}" || ! -x "${afl_fuzz}" ]]; then
    echo "error: AFL++ tools not found under ${afl_root}" >&2
    exit 1
fi

seed_dir=""
out_dir="${ROCFUZZ_AFL_OUT_DIR:-}"
cmake_target=""
use_persistent=1
afl_default_timeout=""

case "${target}" in
rocblas-sgemm)
    cmake_target="rocfuzz_example_rocblas_sgemm_persistent"
    seed_dir="${examples_dir}/seeds/rocblas_sgemm"
    out_dir="${out_dir:-${examples_dir}/afl-out/rocblas-sgemm}"
    export ROCBLAS_USE_HIPBLASLT=0
    ;;
rocfft-c2c)
    cmake_target="rocfuzz_example_rocfft_c2c_persistent"
    seed_dir="${examples_dir}/seeds/rocfft_c2c"
    out_dir="${out_dir:-${examples_dir}/afl-out/rocfft-c2c}"
    ;;
rocrand-uniform)
    cmake_target="rocfuzz_example_rocrand_uniform_persistent"
    seed_dir="${examples_dir}/seeds/rocrand_uniform"
    out_dir="${out_dir:-${examples_dir}/afl-out/rocrand-uniform}"
    ;;
rocsparse-spmv)
    cmake_target="rocfuzz_example_rocsparse_spmv"
    seed_dir="${examples_dir}/seeds/rocsparse_spmv"
    out_dir="${out_dir:-${examples_dir}/afl-out/rocsparse-spmv}"
    use_persistent=0
    afl_default_timeout=5000
    ;;
rocsolver-getrf)
    cmake_target="rocfuzz_example_rocsolver_getrf_persistent"
    seed_dir="${examples_dir}/seeds/rocsolver_getrf"
    out_dir="${out_dir:-${examples_dir}/afl-out/rocsolver-getrf}"
    export ROCBLAS_USE_HIPBLASLT=0
    ;;
miopen-activation)
    cmake_target="rocfuzz_example_miopen_activation_persistent"
    seed_dir="${examples_dir}/seeds/miopen_activation"
    out_dir="${out_dir:-${examples_dir}/afl-out/miopen-activation}"
    ;;
*)
    echo "error: unknown target '${target}'" >&2
    usage >&2
    exit 1
    ;;
esac

cmake --build "${rocjitsu_dir}/build" --target rocjitsu_afl_preload -j "${jobs}"
cmake -S "${examples_dir}" -B "${build_dir}" -G Ninja \
    -DCMAKE_CXX_COMPILER="${afl_cxx}" \
    -DROCFUZZ_AFL_DBI_BUILD="$(dirname -- "${preload}")" \
    -DROCFUZZ_AFL_PRELOAD="${preload}"
cmake --build "${build_dir}" --target "${cmake_target}" -j "${jobs}"

mkdir -p "${out_dir}"

export AFL_AUTORESUME="${AFL_AUTORESUME:-1}"
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES="${AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES:-1}"
export AFL_SKIP_CPUFREQ="${AFL_SKIP_CPUFREQ:-1}"
export LD_PRELOAD="${preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
if [[ "${use_persistent}" == "1" ]]; then
    export ROCJITSU_AFL_PERSISTENT=1
    export ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
fi
if [[ "${ROCFUZZ_REQUIRE_DEVICE_EDGES:-0}" == "1" ]]; then
    export ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
fi

if [[ -n "${afl_default_timeout}" ]]; then
    has_timeout=0
    for arg in "${afl_args[@]}"; do
        if [[ "${arg}" == "-t" || "${arg}" == -t* ]]; then
            has_timeout=1
            break
        fi
    done
    if [[ "${has_timeout}" == "0" ]]; then
        afl_args=("-t" "${afl_default_timeout}" "${afl_args[@]}")
    fi
fi

echo "rocfuzz: target=${target}"
echo "rocfuzz: seeds=${seed_dir}"
echo "rocfuzz: output=${out_dir}"
echo "rocfuzz: binary=${build_dir}/${cmake_target}"

set +e
"${afl_fuzz}" "${afl_args[@]}" -i "${seed_dir}" -o "${out_dir}" -- \
    "${build_dir}/${cmake_target}"
afl_status=$?
set -e

if [[ -n "${ROCFUZZ_AFL_REPORT:-}" ]]; then
    env -u LD_PRELOAD "${script_dir}/summarize-afl-campaign.py" "${out_dir}" \
        --target "${target}" \
        --examples-build "${examples_dir}/build" \
        --output "${ROCFUZZ_AFL_REPORT}"
    echo "rocfuzz: report=${ROCFUZZ_AFL_REPORT}"
fi

exit "${afl_status}"
