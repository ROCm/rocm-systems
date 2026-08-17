#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Download, build, and activate a specific MPI implementation for RHEL CI.
# Usage: setup-mpi.sh <openmpi|mpich>
#
# When GITHUB_PATH / GITHUB_ENV are set (GitHub Actions), exports are written
# there so subsequent workflow steps pick up the chosen MPI.

set -euo pipefail

MPI_IMPL="${1:-}"
if [[ -z "${MPI_IMPL}" ]]; then
    echo "Usage: ${0} <openmpi|mpich>" >&2
    exit 1
fi

# OpenMPI version and download URL, derived from OPENMPI_VERSION.
OPENMPI_VERSION="${OPENMPI_VERSION:-5.0.10}"
OPENMPI_SERIES="${OPENMPI_VERSION%.*}" # Extract the major.minor version from the version string.
OPENMPI_URL="${OPENMPI_URL:-https://download.open-mpi.org/release/open-mpi/v${OPENMPI_SERIES}/openmpi-${OPENMPI_VERSION}.tar.gz}"

# MPICH version and download URL, derived from MPICH_VERSION.
MPICH_VERSION="${MPICH_VERSION:-5.0.1}"
MPICH_URL="${MPICH_URL:-https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz}"

# MPI installation directories, defaulting to /opt/ompi and /opt/mpich, overridden by the MPI_ROOT environment variable.
OPENMPI_ROOT="${MPI_ROOT:-/opt/ompi}"
MPICH_ROOT="${MPI_ROOT:-/opt/mpich}"

# Number of jobs to use for building.
NJOBS="${NJOBS:-$(nproc)}"

# Detect the MPI vendor of the active mpicc by checking for a vendor-unique
# binary (ompi_info / mpichversion) next to it. This CI image ships a system
# OpenMPI RPM on PATH, so a plain `command -v ompi_info` would always match
# regardless of which implementation was actually built; scoping the check to
# mpicc's own directory ties detection to the mpicc actually in use.
# Compiler-flag output (mpicc --showme:compile / -show) is not a reliable
# alternative either, since it only reflects the install prefix (e.g.
# "-I/opt/ompi/include"), which does not necessarily contain the vendor name.
detect_mpi_vendor() {
    local mpicc mpi_bindir
    mpicc="$(command -v mpicc || true)"
    if [[ -z "${mpicc}" ]]; then
        echo "unknown"
        return
    fi

    mpi_bindir="$(dirname "${mpicc}")"
    if [[ -x "${mpi_bindir}/ompi_info" ]]; then
        echo "openmpi"
    elif [[ -x "${mpi_bindir}/mpichversion" ]]; then
        echo "mpich"
    else
        echo "unknown"
    fi
}

# Export the MPI environment variables to the GITHUB_PATH and GITHUB_ENV files.
export_mpi_env() {
    local prefix="${1}"
    if [[ -n "${GITHUB_PATH:-}" ]]; then
        echo "${prefix}/bin" >> "${GITHUB_PATH}"
    fi
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        echo "MPI_ROOT=${prefix}" >> "${GITHUB_ENV}"
        echo "LD_LIBRARY_PATH=${prefix}/lib:${LD_LIBRARY_PATH:-}" >> "${GITHUB_ENV}"
    fi
    export MPI_ROOT="${prefix}"
    export PATH="${prefix}/bin:${PATH}"
    export LD_LIBRARY_PATH="${prefix}/lib:${LD_LIBRARY_PATH:-}"
}

# Download and extract a tarball to a temporary directory.
download_and_extract() {
    local url="${1}"
    local dest="${2}"
    mkdir -p "${dest}"
    curl -fsSL "${url}" | tar -xz -C "${dest}"
}

# Download, build, and install OpenMPI.
build_openmpi() {
    local prefix="${OPENMPI_ROOT}"
    if [[ -x "${prefix}/bin/mpicc" ]]; then
        echo "Using existing OpenMPI install at ${prefix}"
        export_mpi_env "${prefix}"
        return
    fi

    local build_dir
    build_dir="$(mktemp -d)"

    echo "Downloading OpenMPI ${OPENMPI_VERSION} from ${OPENMPI_URL}"
    download_and_extract "${OPENMPI_URL}" "${build_dir}"

    mkdir -p "${prefix}"
    pushd "${build_dir}/openmpi-${OPENMPI_VERSION}" >/dev/null
    ./configure \
        --prefix="${prefix}" \
        --disable-mpi-fortran \
        --with-hwloc=internal \
        --with-libevent=internal
    make -j"${NJOBS}"
    make install
    popd >/dev/null
    rm -rf "${build_dir}"

    export_mpi_env "${prefix}"
}

# Download, build, and install MPICH.
build_mpich() {
    local prefix="${MPICH_ROOT}"
    if [[ -x "${prefix}/bin/mpicc" ]]; then
        echo "Using existing MPICH install at ${prefix}"
        export_mpi_env "${prefix}"
        return
    fi

    local build_dir
    build_dir="$(mktemp -d)"

    echo "Downloading MPICH ${MPICH_VERSION} from ${MPICH_URL}"
    download_and_extract "${MPICH_URL}" "${build_dir}"

    mkdir -p "${prefix}"
    pushd "${build_dir}/mpich-${MPICH_VERSION}" >/dev/null
    ./configure \
        --prefix="${prefix}" \
        --disable-fortran
    make -j"${NJOBS}"
    make install
    popd >/dev/null
    rm -rf "${build_dir}"

    export_mpi_env "${prefix}"
}

# Build the chosen MPI implementation.
case "${MPI_IMPL}" in
    openmpi)
        build_openmpi
        ;;
    mpich)
        build_mpich
        ;;
    *)
        echo "Unsupported MPI implementation: ${MPI_IMPL}" >&2
        echo "Expected: openmpi or mpich" >&2
        exit 1
        ;;
esac

# Verify that the MPI implementation is correctly installed.
mpicc="$(command -v mpicc)"
echo "mpicc: ${mpicc}"
"${mpicc}" --version

# Detect the MPI vendor from the mpicc command.
vendor="$(detect_mpi_vendor)"
echo "Detected MPI vendor: ${vendor}"

# Verify that the detected MPI vendor matches the chosen MPI implementation.
if [[ "${vendor}" != "${MPI_IMPL}" ]]; then
    echo "Error: expected ${MPI_IMPL}, but detected ${vendor}" >&2
    exit 1
fi
