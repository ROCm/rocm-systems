#!/bin/bash -e

: ${EXTRA_ARGS:=""}
: ${EXTRA_TAGS:=""}
: ${BUILD_DIR:=build-release}
: ${VERSION:=0.0.4}
: ${ROCM_VERSION:=4.5.0}
: ${NJOBS:=8}
: ${DISTRO:=""}
: ${LTO:="ON"}

if [ -z "${DISTRO}" ]; then
    DISTRO=$(lsb_release -i | awk '{print $NF}')-$(lsb_release -r | awk '{print $NF}')
fi

STANDARD_ARGS="-DCPACK_GENERATOR=STGZ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF -DOMNITRACE_MAX_THREADS=2048 -DOMNITRACE_BUILD_TESTING=OFF -DOMNITRACE_BUILD_EXAMPLES=OFF -DOMNITRACE_USE_MPI_HEADERS=ON -DOMNITRACE_USE_OMPT=ON -DOMNITRACE_CPACK_SYSTEM_NAME=${DISTRO} -DOMNITRACE_ROCM_VERSION=${ROCM_VERSION} -DOMNITRACE_BUILD_LTO=${LTO} -DTIMEMORY_USE_LIBUNWIND=ON -DTIMEMORY_BUILD_LIBUNWIND=ON -DTIMEMORY_BUILD_PORTABLE=ON"
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

build-and-package()
{
    local DIR=${1}
    shift
    cmake -B ${BUILD_DIR}/${DIR} -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}/${DIR}/install-release ${STANDARD_ARGS} $@ .
    cmake --build ${BUILD_DIR}/${DIR} --target all --parallel ${NJOBS}
    pushd ${BUILD_DIR}/${DIR}
    rm -f *.sh *.deb *.rpm
    cpack -G STGZ
    cpack -G DEB -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/omnitrace
    cpack -G RPM -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/omnitrace
    popd
    cp ${BUILD_DIR}/${DIR}/omnitrace-${VERSION}-*.sh ${BUILD_DIR}/
    cp ${BUILD_DIR}/${DIR}/omnitrace_${VERSION}-*.deb ${BUILD_DIR}/
    cp ${BUILD_DIR}/${DIR}/omnitrace-${VERSION}-*.rpm ${BUILD_DIR}/
}

build-and-package ${DISTRO}-core -DDYNINST_USE_OpenMP=OFF -DOMNITRACE_USE_HIP=OFF
# build-and-package ${DISTRO}-rocm-${ROCM_VERSION} -DOMNITRACE_USE_HIP=ON -DDYNINST_USE_OpenMP=ON
# build-and-package ${DISTRO}-rocm-${ROCM_VERSION}-papi -DOMNITRACE_USE_HIP=ON -DDYNINST_USE_OpenMP=ON -DOMNITRACE_USE_PAPI=ON

# build-and-package ${DISTRO}-rocm-${ROCM_VERSION}-papi-openmpi -DOMNITRACE_USE_HIP=ON -DDYNINST_USE_OpenMP=ON -DOMNITRACE_USE_PAPI=ON -DOMNITRACE_USE_MPI=ON
