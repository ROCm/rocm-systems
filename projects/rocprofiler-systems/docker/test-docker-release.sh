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

    echo ""
    print_default_option() { printf "    --%-20s %-24s     %s (default: %s)\n" "${1}" "${2}" "${3}" "$(tolower ${4})"; }
    print_default_option distro "[ubuntu|opensuse]" "OS distribution" "${DISTRO}"
    print_default_option versions "[VERSION] [VERSION...]" "Ubuntu or OpenSUSE release" "${VERSIONS}"
    print_default_option rocm-versions "[VERSION] [VERSION...]" "ROCm versions" "${ROCM_VERSIONS}"
    print_default_option user "[USERNAME]" "DockerHub username" "${USER}"

    echo ""
    echo "Usage: ${BASH_SOURCE[0]} <OPTIONS> -- <test-release.sh OPTIONS>"
    echo "  e.g:"
    echo "       ${BASH_SOURCE[0]} --distro ubuntu --versions 20.04 --rocm-versions 5.0 5.1 -- --stgz /path/to/stgz/installer"
}

send-error()
{
    echo -e "\nError: ${@}"
    exit 1
}

verbose-run()
{
    echo -e "\n### Executing \"${@}\"... ###\n"
    exec "${@}"
}

test-release()
{
    CONTAINER=${1}
    shift
    local DOCKER_ARGS=""
    tty -s && DOCKER_ARGS="-it" || DOCKER_ARGS=""
    verbose-run docker run ${DOCKER_ARGS} --rm -v ${PWD}:/home/rocprofiler-systems ${CONTAINER} /home/rocprofiler-systems/scripts/test-release.sh ${@}
}

reset-last()
{
    last() { send-error "Unsupported argument :: ${1}"; }
}

reset-last

: ${USER:=$(whoami)}
: ${DISTRO:=ubuntu}
: ${VERSIONS:=22.04 20.04}
: ${ROCM_VERSIONS:=6.2}

n=0
while [[ $# -gt 0 ]]
do
    case "${1}" in
        -h|--help)
            usage
            exit 0
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
        "--rocm-versions")
            shift
            ROCM_VERSIONS=${1}
            last() { ROCM_VERSIONS="${ROCM_VERSIONS} ${1}"; }
            ;;
        --user|-u)
            shift
            USER=${1}
            reset-last
            ;;
        "--")
            shift
            SCRIPT_ARGS=${@}
            break
            ;;
        *)
            last ${1}
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
        test-release ${USER}/rocprofiler-systems:release-base-${TAG}-rocm-${ROCM_VERSION} ${SCRIPT_ARGS}
    done
done
