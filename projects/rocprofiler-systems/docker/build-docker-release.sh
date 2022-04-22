#!/usr/bin/env bash

if [ ! -f CMakeLists.txt ]; then
    if [ ! -f ../CMakeLists.txt ]; then
        echo "Error! Execute script from source directory"
        exit 1
    else
        cd ..
    fi
fi

set -e

send-error()
{
    echo -e "\nError: ${@}"
    exit 1
}

verbose-run()
{
    echo -e "\n\n### Executing \"${@}\"... ###\n"
    eval $@
}

build-release()
{
    CONTAINER=$1
    OS=$2
    ROCM_VERSION=$3
    CODE_VERSION=$4
    shift
    shift
    shift
    shift
    verbose-run docker run -it --rm -v ${PWD}:/home/omnitrace --env DISTRO=${OS} --env ROCM_VERSION=${ROCM_VERSION} --env VERSION=${CODE_VERSION} --env PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\" ${CONTAINER} /home/omnitrace/scripts/build-release.sh ${@}
}

: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}
: ${ROCM_VERSIONS:=5.0 4.5 4.3}
: ${MPI:=0}
: ${PYTHON_VERSIONS:=6 7 8 9}

n=0
while [[ $# -gt 0 ]]
do
    case "${1}" in
        "--distro")
            shift
            DISTRO=${1}
            ;;
        "--versions")
            shift
            VERSIONS=${1}
            ;;
        "--rocm-versions")
            shift
            ROCM_VERSIONS=${1}
            ;;
        "--python-versions")
            shift
            PYTHON_VERSIONS=${1}
            ;;
        "--")
            shift
            SCRIPT_ARGS=${@}
            break
            ;;
        *)
            send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
            ;;
    esac
    n=$((${n} + 1))
    shift
done

CODE_VERSION=$(cat VERSION)

for VERSION in ${VERSIONS}
do
    TAG=${DISTRO}-${VERSION}
    for ROCM_VERSION in ${ROCM_VERSIONS}
    do
        build-release jrmadsen/omnitrace-${TAG}-rocm-${ROCM_VERSION} ${DISTRO}-${VERSION} ${ROCM_VERSION} ${CODE_VERSION} ${SCRIPT_ARGS}
    done
done
