#!/usr/bin/env bash
# Build rocSHMEM (all_backends: IPC + SDMA) from the in-tree projects/rocshmem
# and install it into the workspace. Writes .ci-out/rocshmem.env
# (ROCSHMEM_INSTALL_DIR) for the later RCCL / rccl-tests / run stages.
#
# Mirrors the gin Dockerfile's rocSHMEM step (scripts/build_configs/all_backends
# with USE_IPC=ON USE_SDMA=ON). Unlike ROCm/OpenMPI this is rebuilt every run so
# source changes are exercised.
#
# Environment:
#   ROCM_PATH                 ROCm tree to build against (REQUIRED)
#   MPI_HOME                  OpenMPI install (REQUIRED)
#   GPU_TARGETS               amdgpu arch (default: gfx950)
#   RCCL_BUILD_JOBS           Parallel build jobs (default: $(nproc))
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (rocm-systems checkout)

set -euxo pipefail

GPU_TARGETS="${GPU_TARGETS:-gfx950}"
build_jobs="${RCCL_BUILD_JOBS:-$(nproc)}"

: "${ROCM_PATH:?build-rocshmem.sh: ROCM_PATH must be set (source .ci-out/rocm.env first)}"
: "${MPI_HOME:?build-rocshmem.sh: MPI_HOME must be set (run build-ompi.sh first)}"

WORKDIR="${RCCL_DEVICE_API_WORKDIR:-${WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
fi

src_dir="${WORKDIR}/projects/rocshmem"
[[ -d "${src_dir}" ]] || { echo "ERROR: ${src_dir} not found (sparse checkout?)" >&2; exit 1; }

install_dir="${WORKDIR}/.ci-out/rocshmem"
build_dir="${src_dir}/build"
env_out="${WORKDIR}/.ci-out/rocshmem.env"
mkdir -p "${WORKDIR}/.ci-out"

export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${MPI_HOME}/bin:${PATH}"

rm -rf "${build_dir}" "${install_dir}"
mkdir -p "${build_dir}"

(
  cd "${build_dir}"
  INSTALL_PREFIX="${install_dir}" \
  BUILD_TYPE=Release \
  "${src_dir}/scripts/build_configs/all_backends" \
      -DMPI_ROOT="${MPI_HOME}" \
      -DUSE_EXTERNAL_MPI=OFF \
      -DGPU_TARGETS="${GPU_TARGETS}" \
      -DUSE_IPC=ON -DUSE_SDMA=ON
)

# rocSHMEM's install location for the functional-tests binary moves between
# versions (bin/ -> share/rocshmem/); locate it instead of assuming a path.
tests_bin=""
for d in "${install_dir}/share/rocshmem" "${install_dir}/bin"; do
  if [[ -x "${d}/rocshmem_functional_tests" ]]; then
    tests_bin="${d}/rocshmem_functional_tests"
    break
  fi
done
if [[ -z "${tests_bin}" ]]; then
  echo "ERROR: rocSHMEM install at ${install_dir} has no rocshmem_functional_tests" >&2
  ls -la "${install_dir}"/{share/rocshmem,bin} 2>/dev/null >&2 || true
  exit 1
fi
tests_bin_dir="$(dirname "${tests_bin}")"

{
  printf 'export ROCSHMEM_INSTALL_DIR=%q\n' "${install_dir}"
  printf 'export ROCSHMEM_TESTS_BIN_DIR=%q\n' "${tests_bin_dir}"
} > "${env_out}"
echo "==> Wrote ${env_out} (tests bin dir: ${tests_bin_dir})"
