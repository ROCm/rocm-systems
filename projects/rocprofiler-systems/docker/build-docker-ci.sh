#!/usr/bin/env bash

set -e

: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}
: ${NJOBS=$(nproc)}
: ${ELFUTILS_VERSION:=0.183}

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
        "-j")
            shift
            NJOBS=${1}
            ;;
        "--elfutils-version")
            shift
            ELFUTILS_VERSION=${1}
            ;;
        *)
            send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
            ;;
    esac
    n=$((${n} + 1))
    shift
done

DOCKER_FILE=Dockerfile.${DISTRO}.ci

if [ ! -f ${DOCKER_FILE} ]; then cd docker; fi

if [ ! -f ${DOCKER_FILE} ]; then
    echo "Error! Execute script from source directory"
    exit 1
fi

verbose-run rm -rf ./dyninst-source
verbose-run cp -r ../external/dyninst ./dyninst-source
verbose-run rm -rf ./dyninst-source/{build,install}*

set -e

DISTRO_IMAGE=${DISTRO}

if [ "${DISTRO}" = "opensuse" ]; then DISTRO_IMAGE="opensuse/leap"; fi

for VERSION in ${VERSIONS}
do
    verbose-run docker build . \
        -f ${DOCKER_FILE} \
        --tag jrmadsen/omnitrace-ci:${DISTRO}-${VERSION} \
        --build-arg DISTRO=${DISTRO_IMAGE} \
        --build-arg VERSION=${VERSION} \
        --build-arg NJOBS=${NJOBS} \
        --build-arg ELFUTILS_DOWNLOAD_VERSION=${ELFUTILS_VERSION}
done

verbose-run rm -rf ./dyninst-source
