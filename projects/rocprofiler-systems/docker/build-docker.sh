#!/usr/bin/env bash

: ${ROCM_VERSIONS:="5.0 4.5 4.3"}
: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}

set -e

if [ ! -f Dockerfile ]; then cd docker; fi

for VERSION in ${VERSIONS}
do
    for i in ${ROCM_VERSIONS}
    do
        ROCM_REPO_VERSION=${i}
        if [ "${i}" = "5.0" ]; then ROCM_REPO_VERSION=debian; fi
        if [ "${i}" = "4.1" ]; then ROCM_REPO_DIST="xenial"; fi
        if [ "${i}" = "4.0" ]; then ROCM_REPO_DIST="xenial"; fi
        docker build . --tag jrmadsen/omnitrace-${DISTRO}-${VERSION}-rocm-${i} --build-arg DISTRO=${DISTRO} --build-arg VERSION=${VERSION} --build-arg ROCM_REPO_VERSION=${ROCM_REPO_VERSION} --build-arg ROCM_REPO_DIST=${ROCM_REPO_DIST}
    done
done
