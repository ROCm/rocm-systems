#!/usr/bin/env bash

: ${ROCM_VERSIONS:="4.5 4.3 4.3.1"}
: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}

set -e

if [ ! -f Dockerfile ]; then cd docker; fi

for VERSION in ${VERSIONS}
do
    for i in ${ROCM_VERSIONS}
    do
        docker build . --tag jrmadsen/omnitrace-${DISTRO}-${VERSION}-rocm-${i} --build-arg DISTRO=${DISTRO} --build-arg VERSION=${VERSION} --build-arg ROCM_REPO_VERSION=${i}
    done
done
