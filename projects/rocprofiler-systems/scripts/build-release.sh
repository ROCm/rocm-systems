#!/bin/bash -e

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
: ${PYTHON_VERSIONS:="6 7 8 9 10"}
: ${GENERATORS:="STGZ DEB RPM"}
: ${MPI_IMPL:="openmpi"}
: ${CLEAN:=0}
: ${FRESH:=0}
: ${WITH_CORE:=0}
: ${WITH_MPI:=0}
: ${WITH_ROCM:=0}
: ${WITH_ROCM_MPI:=0}
: ${IS_DOCKER:=0}

if [ -z "${DISTRO}" ]; then
    if [ -f /etc/os-release ]; then
        DISTRO=$(cat /etc/os-release | egrep "^ID=" | sed 's/"/ /g' | sed 's/=/ /g' | sed 's/-/ /g' | awk '{print $2}')-$(cat /etc/os-release | egrep "^VERSION=" | sed 's/"/ /g' | sed 's/=/ /g' | awk '{print $2}')
    else
        DISTRO=$(lsb_release -i | awk '{print $NF}')-$(lsb_release -r | awk '{print $NF}')
    fi
fi

tolower()
{
    echo "$@" | awk -F '\|~\|' '{print tolower($1)}';
}

toupper()
{
    echo "$@" | awk -F '\|~\|' '{print toupper($1)}';
}

usage()
{
    print_option() { printf "    --%-10s %-24s     %s\n" "${1}" "${2}" "${3}"; }
    echo "Options:"
    python_info="(Use '+nopython' to build w/o python, use '+python' to python build with python)"
    print_option core "[+nopython] [+python]" "Core ${python_info}"
    print_option mpi "[+nopython] [+python]" "MPI ${python_info}"
    print_option rocm "[+nopython] [+python]" "ROCm ${python_info}"
    print_option rocm-mpi "[+nopython] [+python]" "ROCm + MPI ${python_info}"
    print_option mpi-impl "[openmpi|mpich]" "MPI implementation"

    echo ""
    print_default_option() { printf "    --%-20s %-14s     %s (default: %s)\n" "${1}" "${2}" "${3}" "$(tolower ${4})"; }
    print_default_option lto "[on|off]" "Enable LTO" "${LTO}"
    print_default_option strip "[on|off]" "Strip libraries" "${STRIP}"
    print_default_option perfetto-tools "[on|off]" "Install perfetto tools" "${PERFETTO_TOOLS}"
    print_default_option static-libgcc "[on|off]" "Build with static libgcc" "${LIBGCC}"
    print_default_option static-libstdcxx "[on|off]" "Build with static libstdc++" "${LIBSTDCXX}"
    print_default_option hidden-visibility "[on|off]" "Build with hidden visibility" "${HIDDEN_VIZ}"
    print_default_option max-threads "N" "Max number of threads supported" "${MAX_THREADS}"
    print_default_option parallel "N" "Number of parallel build jobs" "${NJOBS}"
}

while [[ $# -gt 0 ]]
do
    ARG=${1}
    shift
    VAL=0

    case "${ARG}" in
        --clean)
            CLEAN=1
            continue
            ;;
        --fresh)
            FRESH=1
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
            ;;
        --mpi)
            WITH_MPI=${VAL}
            ;;
        --rocm)
            WITH_ROCM=${VAL}
            ;;
        --rocm-mpi)
            WITH_ROCM_MPI=${VAL}
            ;;
        --mpi-impl)
            MPI_IMPL=${1}
            shift
            ;;
        --lto)
            LTO=$(toupper ${1})
            shift
            ;;
        --static-libgcc)
            LIBGCC=$(toupper ${1})
            shift
            ;;
        --static-libstdcxx)
            LIBSTDCXX=$(toupper ${1})
            shift
            ;;
        --strip)
            STRIP=$(toupper ${1})
            shift
            ;;
        --hidden-visibility)
            HIDDEN_VIZ=$(toupper ${1})
            shift
            ;;
        --perfetto-tools)
            PERFETTO_TOOLS=$(toupper ${1})
            shift
            ;;
        --max-threads)
            MAX_THREADS=${1}
            shift
            ;;
        --parallel)
            NJOBS=${1}
            shift
            ;;
        *)
            echo -e "Error! Unknown option : ${ARG}"
            usage
            exit 1
            ;;
    esac
done

NPROC=$(nproc)
if [ ${NJOBS} -gt ${NPROC} ]; then NJOBS=${NPROC}; fi

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF -DCPACK_GENERATOR=STGZ"
OMNITRACE_GENERAL_ARGS="-DOMNITRACE_CPACK_SYSTEM_NAME=${DISTRO} -DOMNITRACE_ROCM_VERSION=${ROCM_VERSION} -DOMNITRACE_MAX_THREADS=${MAX_THREADS} -DOMNITRACE_STRIP_LIBRARIES=${STRIP} -DOMNITRACE_INSTALL_PERFETTO_TOOLS=${PERFETTO_TOOLS}"
OMNITRACE_BUILD_ARGS="-DOMNITRACE_BUILD_TESTING=OFF -DOMNITRACE_BUILD_EXAMPLES=OFF -DOMNITRACE_BUILD_PAPI=ON -DOMNITRACE_BUILD_LTO=${LTO} -DOMNITRACE_BUILD_HIDDEN_VISIBILITY=${HIDDEN_VIZ} -DOMNITRACE_BUILD_STATIC_LIBGCC=${LIBGCC} -DOMNITRACE_BUILD_STATIC_LIBSTDCXX=${LIBSTDCXX}"
OMNITRACE_USE_ARGS="-DOMNITRACE_USE_MPI_HEADERS=ON -DOMNITRACE_USE_OMPT=ON -DOMNITRACE_USE_PAPI=ON"
TIMEMORY_ARGS="-DTIMEMORY_USE_LIBUNWIND=ON -DTIMEMORY_BUILD_LIBUNWIND=ON -DTIMEMORY_BUILD_PORTABLE=ON"
DYNINST_ARGS="-DOMNITRACE_BUILD_DYNINST=ON -DDYNINST_USE_OpenMP=ON $(echo -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON) -DDYNINST_BOOST_DOWNLOAD_VERSION=${BOOST_VERSION}"
STANDARD_ARGS="${CMAKE_ARGS} ${OMNITRACE_GENERAL_ARGS} ${OMNITRACE_USE_ARGS} ${OMNITRACE_BUILD_ARGS} ${TIMEMORY_ARGS} ${DYNINST_ARGS} ${EXTRA_ARGS}"

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
    verbose-run cat cmake_install.cmake
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
                verbose-run cpack -G DEB -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/omnitrace
                EXT="deb"
                SEP="_"
                DEST="deb"
                ;;
            DEB | RPM)
                verbose-run cpack -G RPM -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/omnitrace
                EXT="rpm"
                SEP="-"
                DEST="rpm"
                ;;
            *)
                echo "Unsupported cpack generator: ${i}"
                continue
                ;;
        esac
        popd
        verbose-run mkdir -p ${BUILD_DIR}/${DEST}
        verbose-run copy-installer ${i} ${BUILD_DIR}/${DEST} ${BUILD_DIR}/${DIR}/omnitrace${SEP}${VERSION}-*.${EXT}
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
    build-and-package-base ${DIR}-python $@ -DOMNITRACE_USE_PYTHON=ON -DOMNITRACE_BUILD_PYTHON=ON -DOMNITRACE_PYTHON_ROOT_DIRS=\"${_PYTHON_ENVS}\"
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

if [ -d /opt/conda/bin ]; then
    export PATH=${PATH}:/opt/conda/bin
    source activate
fi

if [ "${IS_DOCKER}" -ne 0 ]; then git config --global --add safe.directory ${PWD}; fi

build-and-package ${WITH_CORE} ${DISTRO}-core -DOMNITRACE_USE_HIP=OFF
build-and-package ${WITH_MPI} ${DISTRO}-${MPI_IMPL} -DOMNITRACE_USE_HIP=ON
build-and-package ${WITH_ROCM} ${DISTRO}-rocm-${ROCM_VERSION} -DOMNITRACE_USE_HIP=ON
build-and-package ${WITH_ROCM_MPI} ${DISTRO}-rocm-${ROCM_VERSION}-${MPI_IMPL} -DOMNITRACE_USE_HIP=ON -DOMNITRACE_USE_MPI=ON
