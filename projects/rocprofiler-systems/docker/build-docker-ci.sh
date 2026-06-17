#!/usr/bin/env bash

set -e

: ${USER:=$(whoami)}
: ${TYPE:="base"}
: ${DISTRO:=ubuntu}
: ${VERSIONS:=24.04}
: ${NJOBS=$(nproc)}
: ${ELFUTILS_VERSION:=0.188}
: ${PYTHON_VERSIONS:="8 9 10 11 12 13"}
: ${PUSH:=0}
: ${PULL:=--pull}
: ${GPU_TYPE:=""}
: ${GPU_TARBALL:=""}
: ${ROCM_VERSION:=""}

verbose-run()
{
    echo -e "\n### Executing \"${@}\"... ###\n"
    eval $@
}

tolower()
{
    echo "$@" | awk -F '\\|~\\|' '{print tolower($1)}';
}

toupper()
{
    echo "$@" | awk -F '\\|~\\|' '{print toupper($1)}';
}

usage()
{
    print_option() { printf "    --%-20s %-24s     %s\n" "${1}" "${2}" "${3}"; }
    echo "Options:"
    print_option "help -h" "" "This message"
    print_option "push" "" "Push the container to DockerHub when completed"
    print_option "no-pull" "" "Do not pull down most recent base container"

    echo ""
    print_default_option() { printf "    --%-20s %-24s     %s (default: %s)\n" "${1}" "${2}" "${3}" "$(tolower ${4})"; }
    print_default_option distro "[ubuntu|opensuse|rhel|debian]" "OS distribution" "${DISTRO}"
    print_default_option versions "[VERSION] [VERSION...]" "Ubuntu, OpenSUSE, or RHEL release" "${VERSIONS}"
    print_default_option python-versions "[VERSION] [VERSION...]" "Python 3 minor releases" "${PYTHON_VERSIONS}"
    print_default_option rocm-version "[VERSION]" "ROCm version to install (e.g. 7.2)" "${ROCM_VERSION:-none}"
    print_default_option "jobs -j" "[N]" "parallel build jobs" "${NJOBS}"
    print_default_option elfutils-version "[0.183..0.188]" "ElfUtils version" "${ELFUTILS_VERSION}"
    print_default_option user "[USERNAME]" "DockerHub username" "${USER}"
    print_default_option type "[base|gfxXXX]" "Type of image to create" "${TYPE}"
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

reset-last()
{
    last() { send-error "Unsupported argument :: ${1}"; }
}

reset-last

n=0
while [[ $# -gt 0 ]]
do
    case "${1}" in
        -h|--help)
            usage
            exit 0
            ;;
        "--type")
            shift
            TYPE=${1}
            reset-last
            ;;
        "--distro")
            shift
            DISTRO=${1}
            last() { DISTRO="${DISTRO} ${1}"; }
            ;;
        "--versions")
            shift
            VERSIONS=${1}
            last() { VERSIONS="${VERSIONS} ${1}"; }
            ;;
        "--python-versions")
            shift
            PYTHON_VERSIONS=${1}
            last() { PYTHON_VERSIONS="${PYTHON_VERSIONS} ${1}"; }
            ;;
        --jobs|-j)
            shift
            NJOBS=${1}
            reset-last
            ;;
        "--elfutils-version")
            shift
            ELFUTILS_VERSION=${1}
            reset-last
            ;;
        "--gpu-type")
            shift
            GPU_TYPE=${1}
            reset-last
            ;;
        "--gpu-tarball")
            shift
            GPU_TARBALL=${1}
            reset-last
            ;;
        "--rocm-version")
            shift
            ROCM_VERSION=${1}
            reset-last
            ;;
        --user|-u)
            shift
            USER=${1}
            reset-last
            ;;
        "--push")
            PUSH=1
            reset-last
            ;;
        "--no-pull")
            PULL=""
            reset-last
            ;;
        --*)
            reset-last
            last ${1}
            ;;
        *)
            last ${1}
            ;;
    esac
    n=$((${n} + 1))
    shift
done

if [ -n "${ROCM_VERSION}" ] && [ "${TYPE}" = "base" ]; then
    TYPE="rocm-${ROCM_VERSION}"
fi

if [ "${DISTRO}" = "debian" ]; then
    DOCKER_FILE=Dockerfile.ubuntu.ci
else
    DOCKER_FILE=Dockerfile.${DISTRO}.ci
fi

# Build context is projects/rocprofiler-systems/ so that requirements.txt
# is reachable.
if [ ! -f docker/${DOCKER_FILE} ] && [ -f ${DOCKER_FILE} ]; then cd ..; fi

if [ ! -f docker/${DOCKER_FILE} ]; then
    echo "Error! Execute script from source directory"
    exit 1
fi

verbose-run rm -rf ./docker/dyninst-source
verbose-run cp -r ./external/dyninst ./docker/dyninst-source
verbose-run rm -rf ./docker/dyninst-source/{build,install}*

set -e

if [ "${DISTRO}" = "opensuse" ]; then
    DISTRO_IMAGE="opensuse/leap"
elif [ "${DISTRO}" = "rhel" ]; then
    DISTRO_IMAGE="rockylinux/rockylinux"
else
    DISTRO_IMAGE=${DISTRO}
fi

for VERSION in ${VERSIONS}
do
    BASE_IMAGE_ARG=""
    if [ "${DISTRO}" = "ubuntu" ] && [ -n "${ROCM_VERSION}" ]; then
        BASE_IMAGE_ARG="--build-arg BASE_IMAGE=rocm/dev-ubuntu-${VERSION}:${ROCM_VERSION}"
    fi

    verbose-run docker build . \
        ${PULL} \
        -f docker/${DOCKER_FILE} \
        --tag ${USER}/rocprofiler-systems:ci-${TYPE}-${DISTRO}-${VERSION} \
        --build-arg DISTRO=${DISTRO_IMAGE} \
        --build-arg VERSION=${VERSION} \
        ${BASE_IMAGE_ARG} \
        --build-arg NJOBS=${NJOBS} \
        --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\" \
        --build-arg ELFUTILS_DOWNLOAD_VERSION=${ELFUTILS_VERSION} \
        --build-arg GPU_TYPE=${GPU_TYPE} \
        --build-arg GPU_TARBALL=${GPU_TARBALL} \
        --build-arg ROCM_VERSION=${ROCM_VERSION}
done

if [ "${PUSH}" -gt 0 ]; then
    for VERSION in ${VERSIONS}
    do
        verbose-run docker push ${USER}/rocprofiler-systems:ci-${TYPE}-${DISTRO}-${VERSION}
    done
fi

verbose-run rm -rf ./docker/dyninst-source
