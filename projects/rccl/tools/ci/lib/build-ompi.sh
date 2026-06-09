#!/usr/bin/env bash
# Build a GPU-aware OpenMPI (--with-rocm) for the device-API CI job and cache it
# on the shared NFS path. Built against the ROCm tree fetch-rocm.sh resolved, so
# the MPI stack matches the ROCm that RCCL links against.
#
# Result: writes $WORKDIR/.ci-out/ompi.env exporting MPI_HOME so the RCCL /
# rccl-tests / run steps pick up the same install.
#
# The versioned install is reused across runs and only rebuilt when missing or
# when the ROCm it was built --with-rocm against changes.
#
# Environment:
#   ROCM_PATH              ROCm tree to build against (REQUIRED; from fetch-rocm.sh)
#   RCCL_DEVICE_API_CACHE  Persistent cache root (default: /apps/rccl-ci)
#   MPI_HOME_OVERRIDE      Use this existing OpenMPI install; skips building.
#                          An ambient MPI_HOME is intentionally ignored (a stray
#                          one from the node environment must not bypass the build).
#   OMPI_MAJOR_MINOR       OpenMPI major.minor (default: 5.0)
#   OMPI_VERSION           OpenMPI patch version (default: 5.0.9)
#   RCCL_BUILD_JOBS        Parallel build jobs (default: $(nproc))
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)

set -euxo pipefail

OMPI_MAJOR_MINOR="${OMPI_MAJOR_MINOR:-5.0}"
OMPI_VERSION="${OMPI_VERSION:-5.0.9}"
build_jobs="${RCCL_BUILD_JOBS:-$(nproc)}"

: "${ROCM_PATH:?build-ompi.sh: ROCM_PATH must be set (run fetch-rocm.sh first)}"
if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
  echo "ERROR: ROCM_PATH=${ROCM_PATH} has no bin/hipcc" >&2
  exit 1
fi

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

# Rebuild if the install is missing or the ROCm it was built against changed.
rocm_fingerprint="$(cat "${ROCM_PATH}/.stamp" 2>/dev/null || echo "${ROCM_PATH}")"
desired_stamp="ompi=${OMPI_VERSION}
with_rocm:
${rocm_fingerprint}"

if [[ -f "${OMPI_INSTALL_DIR}/lib/libmpi.so" ]] \
   && [[ "$(cat "${OMPI_INSTALL_DIR}/.stamp" 2>/dev/null || true)" == "${desired_stamp}" ]]; then
  echo "==> Reusing cached OpenMPI ${OMPI_VERSION} at ${OMPI_INSTALL_DIR}"
else
  echo "==> Building OpenMPI ${OMPI_VERSION} into ${OMPI_INSTALL_DIR} (--with-rocm=${ROCM_PATH})"
  rm -rf "${OMPI_INSTALL_DIR}"
  mkdir -p "${OMPI_INSTALL_DIR}"
  ompi_src="${CACHE_DIR}/ompi/src-${OMPI_VERSION}"
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
  printf '%s\n' "${desired_stamp}" > "${OMPI_INSTALL_DIR}/.stamp"
fi

printf 'export MPI_HOME=%q\n' "${OMPI_INSTALL_DIR}" > "${env_out}"
echo "==> Wrote ${env_out}"
