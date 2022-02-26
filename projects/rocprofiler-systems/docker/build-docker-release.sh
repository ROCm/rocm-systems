#!/usr/bin/env bash

if [ ! -f CMakeLists.txt ]; then
    echo "Error! Execute script from source directory"
    exit 1
fi

set -e

build-release()
{
    CONTAINER=$1
    ROCM_VERSION=$2
    CODE_VERSION=$3
    MPI=$4
    docker run -it --rm -v ${PWD}:/home/omnitrace --env ROCM_VERSION=${ROCM_VERSION} --env VERSION=${CODE_VERSION} --env MPI=${MPI} ${CONTAINER} /home/omnitrace/scripts/build-release.sh
}

: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}
: ${ROCM_VERSIONS:=5.0 4.5 4.3}
: ${MPI:=0}

CODE_VERSION=$(cat VERSION)

for VERSION in ${VERSIONS}
do
    TAG=${DISTRO}-${VERSION}
    for ROCM_VERSION in ${ROCM_VERSIONS}
    do
        build-release jrmadsen/omnitrace-${TAG}-rocm-${ROCM_VERSION} ${ROCM_VERSION} ${CODE_VERSION} ${MPI}
    done
done
