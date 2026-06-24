#!/usr/bin/env bash
# Build a GPU-aware OpenMPI (--with-rocm) against ROCM_PATH and cache it on the
# shared NFS path. The install is reusable regardless of which ROCm built it, so
# it is rebuilt only when missing (mpirun + libmpi.so present => skip). Writes
# .ci-out/ompi.env (MPI_HOME) for the later RCCL / rccl-tests / run stages.
#
# Environment:
#   ROCM_PATH              ROCm tree to build against (REQUIRED only on a rebuild)
#   RCCL_DEVICE_API_CACHE  Persistent cache root (default: /apps/rccl-ci)
#   MPI_HOME_OVERRIDE      Use this existing install; skips the build.
#                          Ambient MPI_HOME is never trusted.
#   OMPI_MAJOR_MINOR       OpenMPI major.minor (default: 5.0)
#   OMPI_VERSION           OpenMPI patch version (default: 5.0.9)
#   RCCL_BUILD_JOBS        Parallel build jobs (default: $(nproc))
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)

set -euxo pipefail

OMPI_MAJOR_MINOR="${OMPI_MAJOR_MINOR:-5.0}"
OMPI_VERSION="${OMPI_VERSION:-5.0.9}"
build_jobs="${RCCL_BUILD_JOBS:-$(nproc)}"

WORKDIR="${RCCL_DEVICE_API_WORKDIR:-${WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
fi

CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"
downloads="${CACHE_DIR}/downloads"
if ! mkdir -p "${CACHE_DIR}/ompi" "${downloads}"; then
  echo "ERROR: cannot create cache dirs under ${CACHE_DIR} (is /apps writable?)" >&2
  exit 1
fi

env_out="${WORKDIR}/.ci-out/ompi.env"
mkdir -p "${WORKDIR}/.ci-out"

# Explicit override only (not an ambient MPI_HOME, which a module load may set).
if [[ -n "${MPI_HOME_OVERRIDE:-}" ]]; then
  mpi_home="${MPI_HOME_OVERRIDE}"
  if [[ ! -f "${mpi_home}/lib/libmpi.so" ]]; then
    echo "ERROR: MPI_HOME_OVERRIDE=${mpi_home} has no lib/libmpi.so" >&2
    exit 1
  fi
  echo "==> Using MPI_HOME_OVERRIDE=${mpi_home}"
  printf 'export MPI_HOME=%q\n' "${mpi_home}" > "${env_out}"
  exit 0
fi

OMPI_INSTALL_DIR="${CACHE_DIR}/ompi/install/${OMPI_VERSION}"

# Reusable regardless of which ROCm built it; just check the runtime bits exist.
ompi_cached() {
  [[ -x "${OMPI_INSTALL_DIR}/bin/mpirun" ]] \
    && [[ -f "${OMPI_INSTALL_DIR}/lib/libmpi.so" ]]
}

do_build_ompi() {
  : "${ROCM_PATH:?build-ompi.sh: ROCM_PATH must be set to build OpenMPI (run fetch-rocm.sh first)}"
  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: ROCM_PATH=${ROCM_PATH} has no bin/hipcc" >&2
    exit 1
  fi
  echo "==> Building OpenMPI ${OMPI_VERSION} into ${OMPI_INSTALL_DIR} (--with-rocm=${ROCM_PATH})"
  rm -rf "${OMPI_INSTALL_DIR}"
  mkdir -p "${OMPI_INSTALL_DIR}"
  ompi_src="${CACHE_DIR}/ompi/src-${OMPI_VERSION}.$$"
  rm -rf "${ompi_src}"; mkdir -p "${ompi_src}"
  ompi_tar="${downloads}/openmpi-${OMPI_VERSION}.tar.gz"
  [[ -f "${ompi_tar}" ]] || wget -q -O "${ompi_tar}" \
    "https://download.open-mpi.org/release/open-mpi/v${OMPI_MAJOR_MINOR}/openmpi-${OMPI_VERSION}.tar.gz"
  tar -zxf "${ompi_tar}" -C "${ompi_src}" --strip-components=1
  (
    cd "${ompi_src}"
    ./configure --prefix="${OMPI_INSTALL_DIR}" \
        --with-rocm="${ROCM_PATH}" \
        --disable-oshmem \
        --disable-mpi-fortran \
        --enable-orterun-prefix-by-default
    make -j"${build_jobs}"
    make install
  )
  rm -rf "${ompi_src}"
  printf 'ompi=%s\nbuilt_with_rocm=%s\n' \
    "${OMPI_VERSION}" "${ROCM_PATH}" > "${OMPI_INSTALL_DIR}/.stamp"
}

# Build on cache miss only. A per-version lock serializes parallel jobs so they
# can't build/install into the same path at once; the inner re-check lets the
# loser reuse what the winner produced.
if ompi_cached; then
  echo "==> Reusing cached OpenMPI ${OMPI_VERSION} at ${OMPI_INSTALL_DIR}"
else
  exec {lock_fd}>"${downloads}/.lock-ompi-${OMPI_VERSION}"
  flock "${lock_fd}"
  if ompi_cached; then
    echo "==> Reusing cached OpenMPI ${OMPI_VERSION} at ${OMPI_INSTALL_DIR}"
  else
    do_build_ompi
  fi
  flock -u "${lock_fd}"
fi

printf 'export MPI_HOME=%q\n' "${OMPI_INSTALL_DIR}" > "${env_out}"
echo "==> Wrote ${env_out}"
