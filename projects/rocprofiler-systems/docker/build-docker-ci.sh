#!/usr/bin/env bash

set -e

if [ ! -f Dockerfile.ci ]; then cd docker; fi

if [ ! -f Dockerfile.ci ]; then
    echo "Error! Execute script from source directory"
    exit 1
fi

rm -rf ./dyninst-source
cp -r ../external/dyninst ./dyninst-source
rm -rf ./dyninst-source/{build,install}*

: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}
: ${NJOBS=$(nproc)}
: ${ELFUTILS_VERSION:=0.183}

set -e

for VERSION in ${VERSIONS}
do
    docker build . \
        -f Dockerfile.ci \
        --tag jrmadsen/omnitrace-ci:${DISTRO}-${VERSION} \
        --build-arg DISTRO=${DISTRO} \
        --build-arg VERSION=${VERSION} \
        --build-arg NJOBS=${NJOBS} \
        --build-arg ELFUTILS_DOWNLOAD_VERSION=${ELFUTILS_VERSION}
done

rm -rf ./dyninst-source
