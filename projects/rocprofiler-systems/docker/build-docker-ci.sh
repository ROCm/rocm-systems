#!/usr/bin/env bash

set -e

: ${USER:=$(whoami)}
: ${TYPE:="base"}
: ${DISTRO:=ubuntu}
: ${VERSIONS:=24.04}
: ${NJOBS=$(nproc)}
: ${ELFUTILS_VERSION:=0.188}
: ${BOOST_VERSION:=1.79.0}
: ${PYTHON_VERSIONS:="6 7 8 9 10 11 12 13"}
: ${PUSH:=0}
: ${PULL:=--pull}
: ${GPU_TYPE:=""}
: ${GPU_TARBALL:=""}

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
    print_default_option "jobs -j" "[N]" "parallel build jobs" "${NJOBS}"
    print_default_option elfutils-version "[0.183..0.188]" "ElfUtils version" "${ELFUTILS_VERSION}"
    print_default_option boost-version "[1.67.0..1.79.0]" "Boost version" "${BOOST_VERSION}"
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
        "--boost-version")
            shift
            BOOST_VERSION=${1}
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

if [ "${DISTRO}" = "debian" ]; then
    DOCKER_FILE=Dockerfile.ubuntu.ci
else
    DOCKER_FILE=Dockerfile.${DISTRO}.ci
fi

if [ ! -f ${DOCKER_FILE} ]; then cd docker; fi

if [ ! -f ${DOCKER_FILE} ]; then
    echo "Error! Execute script from source directory"
    exit 1
fi

verbose-run rm -rf ./dyninst-source
verbose-run cp -r ../external/dyninst ./dyninst-source
verbose-run rm -rf ./dyninst-source/{build,install}*

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
    verbose-run docker build . \
        ${PULL} \
        -f ${DOCKER_FILE} \
        --tag ${USER}/rocprofiler-systems:ci-${TYPE}-${DISTRO}-${VERSION} \
        --build-arg DISTRO=${DISTRO_IMAGE} \
        --build-arg VERSION=${VERSION} \
        --build-arg NJOBS=${NJOBS} \
        --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\" \
        --build-arg ELFUTILS_DOWNLOAD_VERSION=${ELFUTILS_VERSION} \
        --build-arg BOOST_DOWNLOAD_VERSION=${BOOST_VERSION} \
        --build-arg GPU_TYPE=${GPU_TYPE} \
        --build-arg GPU_TARBALL=${GPU_TARBALL} 
done

if [ "${PUSH}" -gt 0 ]; then
    for VERSION in ${VERSIONS}
    do
        verbose-run docker push ${USER}/rocprofiler-systems:ci-${TYPE}-${DISTRO}-${VERSION}
    done
fi

verbose-run rm -rf ./dyninst-source
