#!/bin/bash -e

: ${EXTRA_ARGS:=""}
: ${EXTRA_TAGS:=""}
: ${VERSION:=0.0.4}
: ${ROCM_VERSION:=4.5.0}
: ${NJOBS:=8}

DISTRO=$(lsb_release -i | awk '{print $NF}')-$(lsb_release -r | awk '{print $NF}')

STANDARD_ARGS="-DCPACK_GENERATOR=STGZ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF -DOMNITRACE_MAX_THREADS=2048 -DOMNITRACE_BUILD_TESTING=OFF -DTIMEMORY_USE_LIBUNWIND=ON -DTIMEMORY_BUILD_LIBUNWIND=ON -DTIMEMORY_BUILD_PORTABLE=ON"
STANDARD_ARGS="${STANDARD_ARGS} -DOMNITRACE_BUILD_DYNINST=ON $(echo -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON)"
if [ -n "${EXTRA_ARGS}" ]; then
    STANDARD_ARGS="${STANDARD_ARGS} ${EXTRA_ARGS}"
fi

PACKAGE_BASE_TAG=omnitrace-${VERSION}-${DISTRO}
if [ -n "${EXTRA_TAGS}" ]; then
    PACKAGE_BASE_TAG="${PACKAGE_BASE_TAG}-${EXTRA_TAGS}"
fi

SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
cd $(dirname ${SCRIPT_DIR})
echo -e "Working directory: $(pwd)"

umask 0000

if [ ! -f build-release/${PACKAGE_BASE_TAG}.sh ]; then
    cmake -B build-release/${DISTRO}-core ${STANDARD_ARGS} -DCMAKE_INSTALL_PREFIX=build-release/${DISTRO}-core/install-release -DDYNINST_USE_OpenMP=OFF -DOMNITRACE_USE_MPI_HEADERS=OFF -DOMNITRACE_USE_HIP=OFF .
    cmake --build build-release/${DISTRO}-core --target package --parallel ${NJOBS}
    cp build-release/${DISTRO}-core/omnitrace-${VERSION}-Linux.sh build-release/${PACKAGE_BASE_TAG}.sh
fi

apt-get install -y libopenmpi-dev openmpi-bin libudev-dev

STANDARD_ARGS="${STANDARD_ARGS} -DOMNITRACE_USE_HIP=ON -DOMNITRACE_USE_MPI_HEADERS=ON -DDYNINST_USE_OpenMP=ON"

if [ ! -f build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}.sh ]; then
    cmake -B build-release/${DISTRO}-rocm-${ROCM_VERSION} -DCMAKE_INSTALL_PREFIX=build-release/${DISTRO}-rocm-${ROCM_VERSION}/install-release ${STANDARD_ARGS} .
    cmake --build build-release/${DISTRO}-rocm-${ROCM_VERSION} --target package --parallel ${NJOBS}
    cp build-release/${DISTRO}-rocm-${ROCM_VERSION}/omnitrace-${VERSION}-Linux.sh build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}.sh
fi

STANDARD_ARGS="${STANDARD_ARGS} -DTIMEMORY_USE_PAPI=ON"

if [ ! -f build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}-PAPI.sh ]; then
    cmake -B build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi -DCMAKE_INSTALL_PREFIX=build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi/install-release ${STANDARD_ARGS} .
    cmake --build build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi --target package --parallel ${NJOBS}
    cp build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi/omnitrace-${VERSION}-Linux.sh build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}-PAPI.sh
fi

if [ "${MPI}" -lt  1 ]; then exit 0; fi

STANDARD_ARGS="${STANDARD_ARGS} -DOMNITRACE_USE_MPI=ON"

if [ ! -f build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}-PAPI-OpenMPI.sh ]; then
    cmake -B build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-openmpi -DCMAKE_INSTALL_PREFIX=build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-openmpi/install-release ${STANDARD_ARGS} .
    cmake --build build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-openmpi --target package --parallel ${NJOBS}
    cp build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-openmpi/omnitrace-${VERSION}-Linux.sh build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}-PAPI-OpenMPI.sh
fi

apt-get purge -y libopenmpi-dev openmpi-bin
apt-get install -y libmpich-dev mpich

if [ ! -f build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}-PAPI-MPICH.sh ]; then
    cmake -B build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-mpich -DCMAKE_INSTALL_PREFIX=build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-mpich/install-release ${STANDARD_ARGS} .
    cmake --build build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-mpich --target package --parallel ${NJOBS}
    cp build-release/${DISTRO}-rocm-${ROCM_VERSION}-papi-mpich/omnitrace-${VERSION}-Linux.sh build-release/${PACKAGE_BASE_TAG}-ROCm-${ROCM_VERSION}-PAPI-MPICH.sh
fi
