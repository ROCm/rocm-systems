#!/usr/bin/env bash

if [ ! -f CMakeLists.txt ]; then
    echo "Error! Execute script from source directory"
    exit 1
fi

set -e

build-release()
{
    CONTAINER=$1
    OS=$2
    ROCM_VERSION=$3
    CODE_VERSION=$4
    MPI=$4
    docker run -it --rm -v ${PWD}:/home/omnitrace --env DISTRO=${OS} --env ROCM_VERSION=${ROCM_VERSION} --env VERSION=${CODE_VERSION} --env MPI=${MPI} ${CONTAINER} /home/omnitrace/scripts/build-release.sh
}

: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}
: ${ROCM_VERSIONS:=5.0 4.5 4.3}
: ${MPI:=0}

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
        *)
            if [ "${n}" -eq 0 ]; then
                DISTRO=${1}
            elif [ "${n}" -eq 1 ]; then
                VERSIONS=${1}
            elif [ "${n}" -eq 2 ]; then
                ROCM_VERSIONS=${1}
            else
                send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
            fi
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
        build-release jrmadsen/omnitrace-${TAG}-rocm-${ROCM_VERSION} ${DISTRO}-${VERSION} ${ROCM_VERSION} ${CODE_VERSION} ${MPI}
    done
done
