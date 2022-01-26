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
    docker run -it --rm -v ${PWD}:/home/omnitrace --env ROCM_VERSION=${ROCM_VERSION} --env VERSION=${CODE_VERSION} ${CONTAINER} /home/omnitrace/scripts/build-release.sh
}

: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}

CODE_VERSION=$(cat VERSION)

for VERSION in ${VERSIONS}
do
    TAG=${DISTRO}-${VERSION}
    build-release jrmadsen/omnitrace-${TAG}-rocm-4.5   4.5.0 ${CODE_VERSION}
    build-release jrmadsen/omnitrace-${TAG}-rocm-4.3   4.3.0 ${CODE_VERSION}
    build-release jrmadsen/omnitrace-${TAG}-rocm-4.3.1 4.3.1 ${CODE_VERSION}
done
