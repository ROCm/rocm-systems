#!/bin/bash -e

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

: ${VERBOSE:=0}
: ${EXTRA_ARGS:=""}
: ${BUILD_DIR:=build-release}
: ${VERSION:=0.0.4}
: ${ROCM_VERSION:=4.5.0}
: ${BOOST_VERSION:=1.79.0}
: ${NJOBS:=12}
: ${DISTRO:=""}
: ${LTO:="OFF"}
: ${STRIP:="OFF"}
: ${LIBGCC:="ON"}
: ${LIBSTDCXX:="ON"}
: ${MAX_THREADS:=2048}
: ${PERFETTO_TOOLS:="ON"}
: ${HIDDEN_VIZ:="ON"}
: ${PYTHON_VERSIONS:="6 7 8 9 10 11 12 13"}
: ${GENERATORS:="STGZ DEB RPM"}
: ${MPI_IMPL:="openmpi"}
: ${CLEAN:=0}
: ${FRESH:=0}
: ${WITH_CORE:=0}
: ${WITH_MPI:=0}
: ${WITH_ROCM:=0}
: ${WITH_ROCM_MPI:=0}
: ${IS_DOCKER:=0}
: ${CONDA_ROOT:=/opt/conda}

if [ -z "${DISTRO}" ]; then
    if [ -f /etc/os-release ]; then
        DISTRO=$(cat /etc/os-release | egrep "^ID=" | sed 's/"/ /g' | sed 's/=/ /g' | sed 's/-/ /g' | awk '{print $2}')-$(cat /etc/os-release | egrep "^VERSION=" | sed 's/"/ /g' | sed 's/=/ /g' | awk '{print $2}')
    else
        DISTRO=$(lsb_release -i | awk '{print $NF}')-$(lsb_release -r | awk '{print $NF}')
    fi
fi

tolower()
{
    echo "$@" | awk -F '\\|~\\|' '{print tolower($1)}';
}

toupper()
{
    echo "$@" | awk -F '\\|~\\|' '{print toupper($1)}';
}

usage()
{
    print_option() { printf "    --%-10s %-36s     %s\n" "${1}" "${2}" "${3}"; }
    echo "Options:"
    python_info="(Use '+nopython' to build w/o python, use '+python' to python build with python)"
    print_option core "[+nopython] [+python]" "Core ${python_info}"
    print_option mpi "[+nopython] [+python]" "MPI ${python_info}"
    print_option rocm "[+nopython] [+python]" "ROCm ${python_info}"
    print_option rocm-mpi "[+nopython] [+python]" "ROCm + MPI ${python_info}"
    print_option mpi-impl "[openmpi|mpich]" "MPI implementation"

    echo ""
    print_default_option() { printf "    --%-20s %-26s     %s (default: %s)\n" "${1}" "${2}" "${3}" "$(tolower ${4})"; }
    print_default_option lto "[on|off]" "Enable LTO" "${LTO}"
    print_default_option strip "[on|off]" "Strip libraries" "${STRIP}"
    print_default_option perfetto-tools "[on|off]" "Install perfetto tools" "${PERFETTO_TOOLS}"
    print_default_option static-libgcc "[on|off]" "Build with static libgcc" "${LIBGCC}"
    print_default_option static-libstdcxx "[on|off]" "Build with static libstdc++" "${LIBSTDCXX}"
    print_default_option hidden-visibility "[on|off]" "Build with hidden visibility" "${HIDDEN_VIZ}"
    print_default_option max-threads "N" "Max number of threads supported" "${MAX_THREADS}"
    print_default_option parallel "N" "Number of parallel build jobs" "${NJOBS}"
    print_default_option generators "[STGZ][DEB][RPM][+others]" "CPack generators" "${GENERATORS}"
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

reset-last()
{
    last() { send-error "Unsupported argument :: ${1}"; }
}

reset-last

while [[ $# -gt 0 ]]
do
    ARG=${1}
    shift
    VAL=0

    case "${ARG}" in
        --clean)
            CLEAN=1
            reset-last
            continue
            ;;
        --fresh)
            FRESH=1
            reset-last
            continue
            ;;
    esac

    while [[ $# -gt 0 ]]
    do
        if [ "$1" = "+nopython" ]; then
            VAL=$(( ${VAL} + 1 ))
            shift
        elif [ "$1" = "+python" ]; then
            VAL=$(( ${VAL} + 2 ))
            shift
        else
            break
        fi
    done

    case "${ARG}" in
        ? | -h | --help)
            usage
            exit 0
            ;;
        --core)
            WITH_CORE=${VAL}
            reset-last
            ;;
        --mpi)
            WITH_MPI=${VAL}
            reset-last
            ;;
        --rocm)
            WITH_ROCM=${VAL}
            reset-last
            ;;
        --rocm-mpi)
            WITH_ROCM_MPI=${VAL}
            reset-last
            ;;
        --mpi-impl)
            MPI_IMPL=${1}
            shift
            reset-last
            ;;
        --lto)
            LTO=$(toupper ${1})
            shift
            reset-last
            ;;
        --static-libgcc)
            LIBGCC=$(toupper ${1})
            shift
            reset-last
            ;;
        --static-libstdcxx)
            LIBSTDCXX=$(toupper ${1})
            shift
            reset-last
            ;;
        --strip)
            STRIP=$(toupper ${1})
            shift
            reset-last
            ;;
        --hidden-visibility)
            HIDDEN_VIZ=$(toupper ${1})
            shift
            reset-last
            ;;
        --perfetto-tools)
            PERFETTO_TOOLS=$(toupper ${1})
            shift
            reset-last
            ;;
        --max-threads)
            MAX_THREADS=${1}
            shift
            reset-last
            ;;
        --parallel)
            NJOBS=${1}
            shift
            reset-last
            ;;
        --generators)
            GENERATORS=$(toupper ${1})
            shift
            last() { GENERATORS="${GENERATORS} $(toupper ${1})"; }
            ;;
        *)
            last ${ARG} ${1}
            ;;
    esac
done

NPROC=$(nproc)
if [ ${NJOBS} -gt ${NPROC} ]; then NJOBS=${NPROC}; fi

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF -DCPACK_GENERATOR=STGZ"
ROCPROFSYS_GENERAL_ARGS="-DROCPROFSYS_CPACK_SYSTEM_NAME=${DISTRO} -DROCPROFSYS_ROCM_VERSION=${ROCM_VERSION} -DROCPROFSYS_MAX_THREADS=${MAX_THREADS} -DROCPROFSYS_STRIP_LIBRARIES=${STRIP} -DROCPROFSYS_INSTALL_PERFETTO_TOOLS=${PERFETTO_TOOLS}"
ROCPROFSYS_BUILD_ARGS="-DROCPROFSYS_BUILD_TESTING=OFF -DROCPROFSYS_BUILD_EXAMPLES=OFF -DROCPROFSYS_BUILD_PAPI=ON -DROCPROFSYS_BUILD_LTO=${LTO} -DROCPROFSYS_BUILD_HIDDEN_VISIBILITY=${HIDDEN_VIZ} -DROCPROFSYS_BUILD_STATIC_LIBGCC=${LIBGCC} -DROCPROFSYS_BUILD_STATIC_LIBSTDCXX=${LIBSTDCXX} -DROCPROFSYS_BUILD_RELEASE=ON"
ROCPROFSYS_USE_ARGS="-DROCPROFSYS_USE_MPI_HEADERS=ON -DROCPROFSYS_USE_OMPT=ON -DROCPROFSYS_USE_PAPI=ON"
TIMEMORY_ARGS="-DTIMEMORY_USE_LIBUNWIND=ON -DTIMEMORY_BUILD_LIBUNWIND=ON -DTIMEMORY_BUILD_PORTABLE=ON"
DYNINST_ARGS="-DROCPROFSYS_BUILD_DYNINST=ON -DDYNINST_USE_OpenMP=ON $(echo -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON) -DDYNINST_BOOST_DOWNLOAD_VERSION=${BOOST_VERSION}"
STANDARD_ARGS="${CMAKE_ARGS} ${ROCPROFSYS_GENERAL_ARGS} ${ROCPROFSYS_USE_ARGS} ${ROCPROFSYS_BUILD_ARGS} ${TIMEMORY_ARGS} ${DYNINST_ARGS} ${EXTRA_ARGS}"

SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
cd $(dirname ${SCRIPT_DIR})
echo -e "Working directory: $(pwd)"

umask 0000

verbose-run()
{
    echo -e "\n##### Executing \"${@}\"... #####\n"
    eval $@
}

copy-installer()
{
    TYPE=${1}
    shift
    DEST=${1}
    shift
    if [ -z "${1}" ]; then
        echo "Warning! No cpack installer for ${TYPE} generator"
        return
    else
        for i in ${@}
        do
            verbose-run cp ${i} ${DEST}/
        done
    fi
}

build-and-package-base()
{
    local DIR=${1}
    shift
    if [ "${FRESH}" -gt 0 ]; then
        verbose-run rm -rf ${BUILD_DIR}/${DIR}/*
    fi
    verbose-run cmake -B ${BUILD_DIR}/${DIR} -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}/${DIR}/install-release ${STANDARD_ARGS} $@ .
    if [ "${CLEAN}" -gt 0 ]; then
        verbose-run cmake --build ${BUILD_DIR}/${DIR} --target clean
    fi
    pushd ${BUILD_DIR}/${DIR}
    verbose-run cat CPackConfig.cmake
    if [ "${VERBOSE}" -gt 0 ]; then
        verbose-run cat cmake_install.cmake
    fi
    popd
    verbose-run cmake --build ${BUILD_DIR}/${DIR} --target all --parallel ${NJOBS}
    verbose-run cmake --build ${BUILD_DIR}/${DIR} --target install --parallel ${NJOBS}
    pushd ${BUILD_DIR}/${DIR}
    verbose-run rm -f *.sh *.deb *.rpm
    popd
    for i in ${GENERATORS}
    do
        pushd ${BUILD_DIR}/${DIR}
        case "${i}" in
            STGZ)
                verbose-run cpack -G STGZ
                EXT="sh"
                SEP="-"
                DEST="stgz"
                ;;
            DEB)
                verbose-run cpack -G DEB -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/rocprofiler-systems
                EXT="deb"
                SEP="_"
                DEST="deb"
                ;;
            RPM)
                verbose-run cpack -G RPM -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/rocprofiler-systems
                EXT="rpm"
                SEP="-"
                DEST="rpm"
                ;;
            7Z | 7ZIP)
                verbose-run cpack -G 7Z
                EXT="7z"
                SEP="-"
                DEST="zip"
                ;;
            TBZ2)
                verbose-run cpack -G TBZ2
                EXT="tar.bz2"
                SEP="-"
                DEST="zip"
                ;;
            TGZ)
                verbose-run cpack -G TGZ
                EXT="tar.gz"
                SEP="-"
                DEST="zip"
                ;;
            TXZ)
                verbose-run cpack -G TXZ
                EXT="tar.xz"
                SEP="-"
                DEST="zip"
                ;;
            TZ)
                verbose-run cpack -G TZ
                EXT="tar.Z"
                SEP="-"
                DEST="zip"
                ;;
            TZST)
                verbose-run cpack -G TZST
                EXT="tar.zst"
                SEP="-"
                DEST="zip"
                ;;
            ZIP)
                verbose-run cpack -G ZIP
                EXT="zip"
                SEP="-"
                DEST="zip"
                ;;
            *)
                echo "Unsupported cpack generator: ${i}"
                continue
                ;;
        esac
        popd
        verbose-run mkdir -p ${BUILD_DIR}/${DEST}
        verbose-run copy-installer ${i} ${BUILD_DIR}/${DEST} ${BUILD_DIR}/${DIR}/rocprofiler-systems${SEP}${VERSION}-*.${EXT}
    done
}

build-and-package-python()
{
    local DIR=${1}
    shift
    local _PYTHON_ENVS=""
    for i in ${PYTHON_VERSIONS}
    do
        conda activate py3.${i}
        if [ -z "${_PYTHON_ENVS}" ]; then
            _PYTHON_ENVS="$(dirname $(dirname $(which python)))"
        else
            _PYTHON_ENVS="${_PYTHON_ENVS};$(dirname $(dirname $(which python)))"
        fi
        conda deactivate
    done
    build-and-package-base ${DIR}-python $@ -DROCPROFSYS_USE_PYTHON=ON -DROCPROFSYS_BUILD_PYTHON=ON -DROCPROFSYS_PYTHON_ROOT_DIRS=\"${_PYTHON_ENVS}\"
}

build-and-package()
{
    local VAL=${1}
    shift
    if [ "${VAL}" -eq 1 ]; then
        build-and-package-base ${@}
    elif [ "${VAL}" -eq 2 ]; then
        build-and-package-python ${@}
    elif [ "${VAL}" -eq 3 ]; then
        build-and-package-base ${@}
        build-and-package-python ${@}
    else
        echo -e "Skipping build/package for ${1}"
    fi
}

if [ -d ${CONDA_ROOT}/bin ]; then
    export PATH=${PATH}:${CONDA_ROOT}/bin
    source activate
fi

if [ "${IS_DOCKER}" -ne 0 ]; then git config --global --add safe.directory ${PWD}; fi

verbose-run echo "Build rocprofiler-systems installers with generators: ${GENERATORS}"

build-and-package ${WITH_CORE} ${DISTRO}-core -DROCPROFSYS_USE_ROCM=OFF -DROCPROFSYS_USE_MPI=OFF
build-and-package ${WITH_MPI} ${DISTRO}-${MPI_IMPL} -DROCPROFSYS_USE_ROCM=OFF -DROCPROFSYS_USE_MPI=ON
build-and-package ${WITH_ROCM} ${DISTRO}-rocm-${ROCM_VERSION} -DROCPROFSYS_USE_ROCM=ON -DROCPROFSYS_USE_MPI=OFF
build-and-package ${WITH_ROCM_MPI} ${DISTRO}-rocm-${ROCM_VERSION}-${MPI_IMPL} -DROCPROFSYS_USE_ROCM=ON -DROCPROFSYS_USE_MPI=ON
