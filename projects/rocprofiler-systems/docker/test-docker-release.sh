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

test-release()
{
    CONTAINER=${1}
    shift
    verbose-run docker run --rm -v ${PWD}:/home/omnitrace ${CONTAINER} /home/omnitrace/scripts/test-release.sh ${@}
}

: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}
: ${ROCM_VERSIONS:=5.0 4.5 4.3}

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
        "--")
            shift
            SCRIPT_ARGS=${@}
            break
            ;;
        *)
            send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
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
        test-release jrmadsen/omnitrace-${TAG}-rocm-${ROCM_VERSION} ${SCRIPT_ARGS}
    done
done
