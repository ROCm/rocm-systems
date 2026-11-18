#!/usr/bin/env bash
# Build a UCX-enabled OpenMPI for the ext-plugins suite, ISOLATED from the shared
# build-ompi.sh used by gin/device-api. The ext-tuner tests launch mpirun with
# `--mca pml ucx`, which the shared OpenMPI (built without UCX) does not provide.
#
# Everything here lives under its own cache subtree and install paths
# (${CACHE_DIR}/ext-plugins/...), so it never collides with or changes the
# gin/device-api OpenMPI. Builds are cached and rebuilt only when missing.
# Writes .ci-out/ompi-ucx.env with MPI_HOME plus the UCX runtime lib path.
#
# Environment:
#   ROCM_PATH              ROCm tree to build against (REQUIRED only on a rebuild)
#   RCCL_DEVICE_API_CACHE  Persistent cache root (default: /apps/rccl-ci)
#   UCX_VERSION            UCX version (default: 1.17.0)
#   OMPI_MAJOR_MINOR       OpenMPI major.minor (default: 5.0)
#   OMPI_VERSION           OpenMPI patch version (default: 5.0.9)
#   OMPI_UCX_BUILD_JOBS    Parallel build jobs for these builds. UCX and OpenMPI
#                          are light C compiles (not memory-heavy like RCCL's C++),
#                          so this defaults higher than the RCCL cap:
#                          min(nproc, 64). Overridable via this env var.
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)

set -euxo pipefail

UCX_VERSION="${UCX_VERSION:-1.17.0}"
OMPI_MAJOR_MINOR="${OMPI_MAJOR_MINOR:-5.0}"
OMPI_VERSION="${OMPI_VERSION:-5.0.9}"
if [[ -n "${OMPI_UCX_BUILD_JOBS:-}" ]]; then
  build_jobs="${OMPI_UCX_BUILD_JOBS}"
else
  _np="$(nproc)"
  build_jobs=$(( _np < 64 ? _np : 64 ))
fi

WORKDIR="${RCCL_DEVICE_API_WORKDIR:-${WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
fi

CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"
# Isolated subtree: nothing here is shared with the gin/device-api OpenMPI.
ext_cache="${CACHE_DIR}/ext-plugins"
downloads="${CACHE_DIR}/downloads"
if ! mkdir -p "${ext_cache}/ucx" "${ext_cache}/ompi" "${downloads}"; then
  echo "ERROR: cannot create cache dirs under ${CACHE_DIR} (is /apps writable?)" >&2
  exit 1
fi

env_out="${WORKDIR}/.ci-out/ompi-ucx.env"
mkdir -p "${WORKDIR}/.ci-out"

UCX_INSTALL_DIR="${ext_cache}/ucx/install/${UCX_VERSION}"
# Install dir tagged with the UCX version so a UCX bump forces an OpenMPI rebuild
# and can never be confused with the shared (non-UCX) OpenMPI cache.
OMPI_INSTALL_DIR="${ext_cache}/ompi/install/${OMPI_VERSION}-ucx${UCX_VERSION}"

ucx_cached()  { [[ -f "${UCX_INSTALL_DIR}/lib/libucp.so" ]]; }
ompi_cached() {
  # OpenMPI 5.x links the UCX pml component into libmpi rather than shipping a
  # standalone lib/openmpi/mca_pml_ucx.so, so a file check for that component
  # never succeeds and would rebuild OpenMPI on every job. Gate on the .stamp
  # written only after a successful `--with-ucx` build + install instead.
  [[ -x "${OMPI_INSTALL_DIR}/bin/mpirun" ]] \
    && [[ -f "${OMPI_INSTALL_DIR}/lib/libmpi.so" ]] \
    && [[ -f "${OMPI_INSTALL_DIR}/.stamp" ]]
}

do_build_ucx() {
  echo "==> Building UCX ${UCX_VERSION} into ${UCX_INSTALL_DIR}"
  rm -rf "${UCX_INSTALL_DIR}"; mkdir -p "${UCX_INSTALL_DIR}"
  local src="${ext_cache}/ucx/src-${UCX_VERSION}.$$"
  rm -rf "${src}"; mkdir -p "${src}"
  local tar="${downloads}/ucx-${UCX_VERSION}.tar.gz"
  [[ -f "${tar}" ]] || wget -q -O "${tar}" \
    "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz"
  tar -zxf "${tar}" -C "${src}" --strip-components=1
  (
    cd "${src}"
    # Plain UCX is enough: MPI only uses it for its own bootstrap transport, RCCL
    # drives the GPU collectives itself. Keep the dependency surface minimal.
    ./configure --prefix="${UCX_INSTALL_DIR}" \
        --without-java \
        --disable-doxygen-doc \
        --enable-optimizations
    make -j"${build_jobs}"
    make install
  )
  rm -rf "${src}"
}

do_build_ompi() {
  : "${ROCM_PATH:?build-ompi-ucx.sh: ROCM_PATH must be set (source .ci-out/rocm.env first)}"
  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: ROCM_PATH=${ROCM_PATH} has no bin/hipcc" >&2
    exit 1
  fi
  echo "==> Building OpenMPI ${OMPI_VERSION} (--with-ucx=${UCX_INSTALL_DIR}) into ${OMPI_INSTALL_DIR}"
  rm -rf "${OMPI_INSTALL_DIR}"; mkdir -p "${OMPI_INSTALL_DIR}"
  local src="${ext_cache}/ompi/src-${OMPI_VERSION}.$$"
  rm -rf "${src}"; mkdir -p "${src}"
  local tar="${downloads}/openmpi-${OMPI_VERSION}.tar.gz"
  [[ -f "${tar}" ]] || wget -q -O "${tar}" \
    "https://download.open-mpi.org/release/open-mpi/v${OMPI_MAJOR_MINOR}/openmpi-${OMPI_VERSION}.tar.gz"
  tar -zxf "${tar}" -C "${src}" --strip-components=1
  (
    cd "${src}"
    ./configure --prefix="${OMPI_INSTALL_DIR}" \
        --with-rocm="${ROCM_PATH}" \
        --with-ucx="${UCX_INSTALL_DIR}" \
        --disable-oshmem \
        --disable-mpi-fortran \
        --enable-orterun-prefix-by-default
    make -j"${build_jobs}"
    make install
  )
  rm -rf "${src}"
  printf 'ompi=%s\nucx=%s\nbuilt_with_rocm=%s\n' \
    "${OMPI_VERSION}" "${UCX_VERSION}" "${ROCM_PATH}" > "${OMPI_INSTALL_DIR}/.stamp"
}

# UCX first (OpenMPI links it), each guarded by a per-version lock so parallel
# jobs serialize into the same install path; the inner re-check lets the loser
# reuse what the winner produced.
if ucx_cached; then
  echo "==> Reusing cached UCX ${UCX_VERSION} at ${UCX_INSTALL_DIR}"
else
  exec {lock_fd}>"${downloads}/.lock-ucx-${UCX_VERSION}"
  flock "${lock_fd}"
  if ucx_cached; then
    echo "==> Reusing cached UCX ${UCX_VERSION} at ${UCX_INSTALL_DIR}"
  else
    do_build_ucx
  fi
  flock -u "${lock_fd}"
fi

if ompi_cached; then
  echo "==> Reusing cached UCX-enabled OpenMPI ${OMPI_VERSION} at ${OMPI_INSTALL_DIR}"
else
  exec {lock_fd}>"${downloads}/.lock-ompi-ucx-${OMPI_VERSION}-${UCX_VERSION}"
  flock "${lock_fd}"
  if ompi_cached; then
    echo "==> Reusing cached UCX-enabled OpenMPI ${OMPI_VERSION} at ${OMPI_INSTALL_DIR}"
  else
    do_build_ompi
  fi
  flock -u "${lock_fd}"
fi

{
  printf 'export MPI_HOME=%q\n' "${OMPI_INSTALL_DIR}"
  printf 'export UCX_HOME=%q\n' "${UCX_INSTALL_DIR}"
  # UCX runtime libs must be resolvable when mpirun selects `--mca pml ucx`.
  printf 'export LD_LIBRARY_PATH=%q:%q:${LD_LIBRARY_PATH:-}\n' \
    "${OMPI_INSTALL_DIR}/lib" "${UCX_INSTALL_DIR}/lib"
} > "${env_out}"
echo "==> Wrote ${env_out}"
