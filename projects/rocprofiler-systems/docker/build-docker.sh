#!/usr/bin/env bash

: ${ROCM_VERSIONS:="5.0 4.5 4.3"}
: ${DISTRO:=ubuntu}
: ${VERSIONS:=20.04 18.04}
: ${PYTHON_VERSIONS:="6 7 8 9"}
: ${CI:=""}

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
        "--python-versions")
            shift
            PYTHON_VERSIONS=${1}
            ;;
        *)
            if [ "${n}" -eq 0 ]; then
                DISTRO=${1}
            elif [ "${n}" -eq 1 ]; then
                VERSIONS=${1}
            elif [ "${n}" -eq 2 ]; then
                ROCM_VERSIONS=${1}
            else
                send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
            fi
            ;;
    esac
    n=$((${n} + 1))
    shift
done

DOCKER_FILE="Dockerfile.${DISTRO}"

if [ -n "${CI}" ]; then DOCKER_FILE="${DOCKER_FILE}.ci"; fi
if [ ! -f ${DOCKER_FILE} ]; then cd docker; fi
if [ ! -f ${DOCKER_FILE} ]; then send-error "File \"${DOCKER_FILE}\" not found"; fi

for VERSION in ${VERSIONS}
do
    for i in ${ROCM_VERSIONS}
    do
        if [ "${DISTRO}" = "ubuntu" ]; then
            ROCM_REPO_DIST="ubuntu"
            ROCM_REPO_VERSION=${i}
            case "${i}" in
                5.1*)
                    ROCM_REPO_VERSION="debian"
                    ;;
                4.1* | 4.0*)
                    ROCM_REPO_DIST="xenial"
                    ;;
                *)
                    ;;
            esac
            verbose-run docker build . -f ${DOCKER_FILE} --tag jrmadsen/omnitrace-${DISTRO}-${VERSION}-rocm-${i} --build-arg DISTRO=${DISTRO} --build-arg VERSION=${VERSION} --build-arg ROCM_REPO_VERSION=${ROCM_REPO_VERSION} --build-arg ROCM_REPO_DIST=${ROCM_REPO_DIST} --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\"
        elif [ "${DISTRO}" = "centos" ]; then
            case "${VERSION}" in
                7)
                    RPM_PATH=7.9
                    RPM_TAG=".el7"
                    ;;
                8)
                    RPM_PATH=8.5
                    RPM_TAG=".el7"
                    ;;
                *)
                    send-error "Invalid centos version ${VERSION}. Supported: 7, 8"
            esac
            case "${i}" in
                5.0*)
                    ROCM_RPM=latest/rhel/${RPM_PATH}/amdgpu-install-21.50.50000-1${RPM_TAG}.noarch.rpm
                    ;;
                4.5 | 4.5.2)
                    ROCM_RPM=21.40.2/rhel/${RPM_PATH}/amdgpu-install-21.40.2.40502-1${RPM_TAG}.noarch.rpm
                    ;;
                4.5.1)
                    ROCM_RPM=21.40.1/rhel/${RPM_PATH}/amdgpu-install-21.40.1.40501-1${RPM_TAG}.noarch.rpm
                    ;;
                4.5.0)
                    ROCM_RPM=21.40/rhel/${RPM_PATH}/amdgpu-install-21.40.1.40501-1${RPM_TAG}.noarch.rpm
                    ;;
                *)
                    send-error "Unsupported combination :: ${DISTRO}-${VERSION} + ROCm ${i}"
                    ;;
            esac
            verbose-run docker build . -f ${DOCKER_FILE} --tag jrmadsen/omnitrace-${DISTRO}-${VERSION}-rocm-${i} --build-arg DISTRO=${DISTRO} --build-arg VERSION=${VERSION} --build-arg AMDGPU_RPM=${ROCM_RPM} --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\"
        elif [ "${DISTRO}" = "opensuse" ]; then
            case "${VERSION}" in
                15.*)
                    DISTRO_IMAGE="opensuse/leap"
                    echo "DISTRO_IMAGE: ${DISTRO_IMAGE}"
                    ;;
                *)
                    send-error "Invalid opensuse version ${VERSION}. Supported: 15.x"
                    ;;
            esac
            case "${i}" in
                5.0*)
                    ROCM_RPM=latest/sle/15/amdgpu-install-21.50.50000-1.noarch.rpm
                    ;;
                4.5 | 4.5.2)
                    ROCM_RPM=21.40.2/sle/15/amdgpu-install-21.40.2.40502-1.noarch.rpm
                    ;;
                4.5.1)
                    ROCM_RPM=21.40.1/sle/15/amdgpu-install-21.40.1.40501-1.noarch.rpm
                    ;;
                4.5.0)
                    ROCM_RPM=21.40/sle/15/amdgpu-install-21.40.1.40501-1.noarch.rpm
                    ;;
                *)
                    send-error "Unsupported combination :: ${DISTRO}-${VERSION} + ROCm ${i}"
                ;;
            esac
            verbose-run docker build . -f ${DOCKER_FILE} --tag jrmadsen/omnitrace-${DISTRO}-${VERSION}-rocm-${i} --build-arg DISTRO=${DISTRO_IMAGE} --build-arg VERSION=${VERSION} --build-arg AMDGPU_RPM=${ROCM_RPM} --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\"
        fi
    done
done
