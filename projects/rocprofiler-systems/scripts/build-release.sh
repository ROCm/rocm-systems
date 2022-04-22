#!/bin/bash -e

: ${EXTRA_ARGS:=""}
: ${BUILD_DIR:=build-release}
: ${VERSION:=0.0.4}
: ${ROCM_VERSION:=4.5.0}
: ${NJOBS:=12}
: ${DISTRO:=""}
: ${LTO:="OFF"}
: ${PYTHON_VERSIONS:="6 7 8 9"}
: ${GENERATORS:="STGZ DEB RPM"}
: ${MPI_IMPL:="openmpi"}

if [ -z "${DISTRO}" ]; then
    if [ -f /etc/os-release ]; then
        DISTRO=$(cat /etc/os-release | egrep "^ID=" | sed 's/"/ /g' | sed 's/=/ /g' | sed 's/-/ /g' | awk '{print $2}')-$(cat /etc/os-release | egrep "^VERSION=" | sed 's/"/ /g' | sed 's/=/ /g' | awk '{print $2}')
    else
        DISTRO=$(lsb_release -i | awk '{print $NF}')-$(lsb_release -r | awk '{print $NF}')
    fi
fi

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF -DCPACK_GENERATOR=STGZ"
OMNITRACE_GENERAL_ARGS="-DOMNITRACE_CPACK_SYSTEM_NAME=${DISTRO} -DOMNITRACE_ROCM_VERSION=${ROCM_VERSION} -DOMNITRACE_MAX_THREADS=2048 "
OMNITRACE_BUILD_ARGS="-DOMNITRACE_BUILD_TESTING=OFF -DOMNITRACE_BUILD_EXAMPLES=OFF -DOMNITRACE_BUILD_PAPI=ON -DOMNITRACE_BUILD_LTO=${LTO}"
OMNITRACE_USE_ARGS="-DOMNITRACE_USE_MPI_HEADERS=ON -DOMNITRACE_USE_OMPT=ON -DOMNITRACE_USE_PAPI=OFF"
TIMEMORY_ARGS="-DTIMEMORY_USE_LIBUNWIND=ON -DTIMEMORY_BUILD_LIBUNWIND=ON -DTIMEMORY_BUILD_PORTABLE=ON"
DYNINST_ARGS="${STANDARD_ARGS} -DOMNITRACE_BUILD_DYNINST=ON $(echo -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON)"
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

tolower()
{
    echo "$@" | awk -F '\|~\|' '{print tolower($1)}';
}

build-and-package-base()
{
    local DIR=${1}
    shift
    verbose-run cmake -B ${BUILD_DIR}/${DIR} -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}/${DIR}/install-release ${STANDARD_ARGS} $@ .
    verbose-run cmake --build ${BUILD_DIR}/${DIR} --target all --parallel ${NJOBS}
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
    for i in ${PYTHON_VERSIONS}
    do
        conda activate py3.${i}
        _PYTHON_VERS="${_PYTHON_VERS}3.${i};"
        _PYTHON_DIRS="${_PYTHON_DIRS}$(dirname $(dirname $(which python)));"
        conda deactivate
    done
    build-and-package-base ${DIR}-python $@ -DOMNITRACE_USE_PYTHON=ON -DOMNITRACE_BUILD_PYTHON=ON -DOMNITRACE_PYTHON_VERSIONS=\"${_PYTHON_VERS}\" -DOMNITRACE_PYTHON_ROOT_DIRS=\"${_PYTHON_DIRS}\"
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

: ${WITH_CORE:=0}
: ${WITH_MPI:=0}
: ${WITH_ROCM:=0}
: ${WITH_ROCM_MPI:=0}

usage()
{
    print_option() { printf "    --%-10s %-24s     %s\n" "${1}" "${2}" "${3}"; }
    echo "Options:"
    python_info="(Use '-python' to build w/o python, use '+python' to python build with python)"
    print_option core "[+nopython] [+python]" "Core ${python_info}"
    print_option mpi "[+nopython] [+python]" "MPI ${python_info}"
    print_option rocm "[+nopython] [+python]" "ROCm ${python_info}"
    print_option rocm-mpi "[+nopython] [+python]" "ROCm + MPI ${python_info}"
    print_option mpi-impl "[openmpi|mpich]" "MPI implementation"
}

while [[ $# -gt 0 ]]
do
    ARG=${1}
    shift
    VAL=0

    while [[ $# -gt 0 ]]
    do
        if [ "$1" = "-python" ]; then
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
        *)
            echo -e "Error! Unknown option : ${ARG}"
            usage
            exit 1
            ;;
    esac
done

if [ -d /opt/conda/bin ]; then
    export PATH=${PATH}:/opt/conda/bin
    source activate
fi

build-and-package ${WITH_CORE} ${DISTRO}-core -DDYNINST_USE_OpenMP=OFF -DOMNITRACE_USE_HIP=OFF
build-and-package ${WITH_MPI} ${DISTRO}-${MPI_IMPL} -DOMNITRACE_USE_HIP=ON -DDYNINST_USE_OpenMP=ON
build-and-package ${WITH_ROCM} ${DISTRO}-rocm-${ROCM_VERSION} -DOMNITRACE_USE_HIP=ON -DDYNINST_USE_OpenMP=ON
build-and-package ${WITH_ROCM_MPI} ${DISTRO}-rocm-${ROCM_VERSION}-${MPI_IMPL} -DOMNITRACE_USE_HIP=ON -DDYNINST_USE_OpenMP=ON -DOMNITRACE_USE_MPI=ON
