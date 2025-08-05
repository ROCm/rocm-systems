#!/usr/bin/env bash

set-user-defaults()
{
    : ${USER:=$(whoami)}
    : ${ROCM_VERSIONS:="6.3"}
    : ${DISTRO:=ubuntu}
    : ${VERSIONS:=20.04}
    : ${PYTHON_VERSIONS:="6 7 8 9 10 11 12 13"}
    : ${BUILD_CI:=""}
    : ${PUSH:=0}
    : ${PULL:=--pull}
    : ${RETRY:=3}
    : ${SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")}
}

set-user-defaults

set -e

cd $(dirname ${SCRIPT_DIR})

declare -a MATRIX_DISTROS=()
declare -a MATRIX_VERSIONS=()
declare -a MATRIX_ROCM_VERSIONS=()

load-matrix()
{
    local workflow_file=".github/workflows/containers.yml"
    if [ ! -f "${workflow_file}" ]; then
        echo -e "\n Error: Cannot find ${workflow_file}"
        exit 1
    fi

    # In form os-distro;os-version;rocm-version
    local matrix_data=$(awk '
    /rocprofiler-systems-release:/, /steps:/ {
        if (/- os-distro:/) {
            gsub(/[[:space:]]*- os-distro:[[:space:]]*"/, "")
            gsub(/"/, "")
            distro = $0
        }
        if (/os-version:/) {
            gsub(/[[:space:]]*os-version:[[:space:]]*"/, "")
            gsub(/"/, "")
            version = $0
        }
        if (/rocm-version:/) {
            gsub(/[[:space:]]*rocm-version:[[:space:]]*"/, "")
            gsub(/"/, "")
            rocm = $0
            printf "%s;%s;%s\n", distro, version, rocm
        }
    }
    ' "${workflow_file}")

    while IFS=';' read -r os_distro os_version rocm_version; do
        MATRIX_DISTROS+=("$os_distro")
        MATRIX_VERSIONS+=("$os_version")
        MATRIX_ROCM_VERSIONS+=("$rocm_version")
    done <<< "$matrix_data"
}

validate-distro()
{
    local distro="${1}"

    if [ -n "${distro}" ]; then
        distro=$(tolower "${distro}")
        case "${distro}" in
            ubuntu|opensuse|rhel)
                ;;
            *)
                send-error "Unsupported distribution '${distro}'" "Supported distributions: ubuntu, opensuse, rhel"
                ;;
        esac
    fi
}

show-matrix()
{
    local filter_distro="${1:-}"

    if [ -n "${filter_distro}" ]; then
        validate-distro "${filter_distro}"
        filter_distro=$(tolower "${filter_distro}")
    fi

    echo ""
    if [ -n "${filter_distro}" ]; then
        echo "        Supported ${filter_distro} + ROCm Combinations     "
        echo "   =============================================="
    else
        echo "        Supported OS + ROCm Combinations     "
        echo "   =========================================="
    fi
    echo ""
    echo "   OS Distribution    Version    ROCm Version"
    echo "   ----------------   -------    ------------"

    for i in "${!MATRIX_DISTROS[@]}"; do
        if [[ -z "${filter_distro}" || "${MATRIX_DISTROS[i]}" == "${filter_distro}" ]]; then
            printf "   %-16s   %-9s  %s\n" "${MATRIX_DISTROS[i]}" "${MATRIX_VERSIONS[i]}" "${MATRIX_ROCM_VERSIONS[i]}"
        fi
    done

    echo ""
    echo "ROCm '0.0' means no ROCm installation (CPU-only build)"
    echo ""
    echo "Note: Patch versions are also supported (See: https://repo.radeon.com/amdgpu-install/)"
    echo ""
}

# Cross checks arguments against compatibility matrix (ignores ROCm patch version)
validate-combinations()
{
    # Check OS version combinations
    for VERSION in ${VERSIONS}; do
        VERSION_MAJOR=$(echo ${VERSION} | sed 's/\./ /g' | awk '{print $1}')
        VERSION_MINOR=$(echo ${VERSION} | sed 's/\./ /g' | awk '{print $2}')

        local os_version_valid=0
        for i in "${!MATRIX_DISTROS[@]}"; do
            if [[ "${MATRIX_DISTROS[i]}" == "${DISTRO}" && \
                  "${MATRIX_VERSIONS[i]}" == "${VERSION}" ]]; then
                os_version_valid=1
                break
            fi
        done

        if [ ${os_version_valid} -eq 0 ]; then
            send-error "Unsupported OS version :: ${VERSION}. See compatibility matrix for supported versions."
        fi
    done

    # Check ROCm version combinations
    for VERSION in ${VERSIONS}; do
        for ROCM_VERSION in ${ROCM_VERSIONS}; do
            local valid=0
            ROCM_MAJOR=$(echo ${ROCM_VERSION} | sed 's/\./ /g' | awk '{print $1}')
            ROCM_MINOR=$(echo ${ROCM_VERSION} | sed 's/\./ /g' | awk '{print $2}')
            ROCM_MAJOR_MINOR="${ROCM_MAJOR}.${ROCM_MINOR}"
            if ! ([ "${ROCM_MAJOR_MINOR}" == "0.0" ] && [ "${ROCM_VERSION}" != "0.0" ]); then
                for i in "${!MATRIX_DISTROS[@]}"; do
                    if [[ "${MATRIX_DISTROS[i]}" == "${DISTRO}" && \
                        "${MATRIX_VERSIONS[i]}" == "${VERSION}" && \
                        "${MATRIX_ROCM_VERSIONS[i]}" == "${ROCM_MAJOR_MINOR}" ]]; then
                        valid=1
                        break
                    fi
                done
            fi

            if [ ${valid} -eq 0 ]; then
                send-error "Unsupported combination :: ${DISTRO}-${VERSION} + ROCm ${ROCM_VERSION}. See compatibility matrix for supported versions."
            fi
        done
    done
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
    set-user-defaults
    print_option() { printf "    --%-20s %-24s     %s\n" "${1}" "${2}" "${3}"; }
    echo "Options:"
    print_option "help -h"   "" "This message"
    print_option "no-pull"   "" "Do not pull down most recent base container"
    print_option "matrix -m" "[ubuntu|opensuse|rhel]" "Shows compatibility matrix"

    echo ""
    print_default_option() { printf "    --%-20s %-24s     %s (default: %s)\n" "${1}" "${2}" "${3}" "$(tolower ${4})"; }
    print_default_option distro "[ubuntu|opensuse|rhel]" "OS distribution" "${DISTRO}"
    print_default_option versions "[VERSION] [VERSION...]" "Ubuntu, OpenSUSE, or RHEL release" "${VERSIONS}"
    print_default_option rocm-versions "[VERSION] [VERSION...]" "ROCm versions (format: Major.Minor.Patch, patch defaults to 0 if not specified)" "${ROCM_VERSIONS}"
    print_default_option python-versions "[VERSION] [VERSION...]" "Python 3 minor releases" "${PYTHON_VERSIONS}"
    print_default_option "user -u" "[USERNAME]" "DockerHub username" "${USER}"
    print_default_option "retry -r" "[N]" "Number of attempts to build (to account for network errors)" "${RETRY}"
    print_default_option push "" "Push the image to Dockerhub" ""
    #print_default_option lto "[on|off]" "Enable LTO" "${LTO}"
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

verbose-run()
{
    echo -e "\n### Executing \"${@}\"... ###\n"
    eval "${@}"
}

verbose-build()
{
    echo -e "\n### Executing \"${@}\" a maximum of ${RETRY} times... ###\n"
    for i in $(seq 1 1 ${RETRY})
    do
        set +e
        eval "${@}"
        local RETC=$?
        set -e
        if [ "${RETC}" -eq 0 ]; then
            break
        else
            echo -en "\n### Command failed with error code ${RETC}... "
            if [ "${i}" -ne "${RETRY}" ]; then
                echo -e "Retrying... ###\n"
                sleep 3
            else
                echo -e "Exiting... ###\n"
                exit ${RETC}
            fi
        fi
    done
}

reset-last()
{
    last() { send-error "Unsupported argument :: ${1}"; }
}

reset-last
load-matrix
n=0
while [[ $# -gt 0 ]]
do
    case "${1}" in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--matrix)
            shift
            if [[ $# -gt 0 && ! "${1}" =~ ^-- ]]; then
                show-matrix "${1}"
            else
                show-matrix
            fi
            exit 0
            ;;
        "--distro")
            shift
            DISTRO=$(tolower ${1})
            last() { DISTRO="${DISTRO} $(tolower${1})"; }
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
        "--python-versions")
            shift
            PYTHON_VERSIONS=${1}
            last() { PYTHON_VERSIONS="${PYTHON_VERSIONS} ${1}"; }
            ;;
        --user|-u)
            shift
            USER=${1}
            reset-last
            ;;
        --push)
            PUSH=1
            reset-last
            ;;
        --no-pull)
            PULL=""
            reset-last
            ;;
        --retry|-r)
            shift
            RETRY=${1}
            reset-last
            ;;
        "--*")
            send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
            ;;
        *)
            last ${1}
            ;;
    esac
    n=$((${n} + 1))
    shift
done

# Validate input parameters for os-distros and rocm-versions
validate-distro
validate-combinations

DOCKER_FILE="Dockerfile.${DISTRO}"

if [ "${RETRY}" -lt 1 ]; then
    RETRY=1
fi

if [ -n "${BUILD_CI}" ]; then DOCKER_FILE="${DOCKER_FILE}.ci"; fi
cd docker # Forced since PWD is parent dir
if [ ! -f ${DOCKER_FILE} ]; then send-error "File \"${DOCKER_FILE}\" not found"; fi

for VERSION in ${VERSIONS}
do
    VERSION_MAJOR=$(echo ${VERSION} | sed 's/\./ /g' | awk '{print $1}')
    VERSION_MINOR=$(echo ${VERSION} | sed 's/\./ /g' | awk '{print $2}')
    VERSION_PATCH=$(echo ${VERSION} | sed 's/\./ /g' | awk '{print $3}')
    for ROCM_VERSION in ${ROCM_VERSIONS}
    do
        ROCM_MAJOR=$(echo ${ROCM_VERSION} | sed 's/\./ /g' | awk '{print $1}')
        ROCM_MINOR=$(echo ${ROCM_VERSION} | sed 's/\./ /g' | awk '{print $2}')
        ROCM_PATCH=$(echo ${ROCM_VERSION} | sed 's/\./ /g' | awk '{print $3}')
        if [ "${ROCM_PATCH}" = "0" ] || [ -z "${ROCM_PATCH}" ]; then
            CONTAINER=${USER}/rocprofiler-systems:release-base-${DISTRO}-${VERSION}-rocm-${ROCM_MAJOR}.${ROCM_MINOR}
        else
            CONTAINER=${USER}/rocprofiler-systems:release-base-${DISTRO}-${VERSION}-rocm-${ROCM_VERSION}
        fi
        if [ "${DISTRO}" = "ubuntu" ]; then
            case "${VERSION}" in
                24.04)
                    ROCM_REPO_DIST="noble"
                    ;;
                22.04)
                    ROCM_REPO_DIST="jammy"
                    ;;
                20.04)
                    ROCM_REPO_DIST="focal"
                    ;;
                *)
                    ;;
            esac
            verbose-build docker build . ${PULL} --progress plain -f ${DOCKER_FILE} --tag ${CONTAINER} --build-arg DISTRO=${DISTRO} --build-arg VERSION=${VERSION} --build-arg ROCM_VERSION=${ROCM_VERSION} --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\"
        elif [ "${DISTRO}" = "rhel" ]; then
            # use Rocky Linux as a base image for RHEL builds
            DISTRO_BASE_IMAGE=rockylinux/rockylinux
            verbose-build docker build . ${PULL} --progress plain -f ${DOCKER_FILE} --tag ${CONTAINER} --build-arg DISTRO=${DISTRO_BASE_IMAGE} --build-arg VERSION=${VERSION} --build-arg ROCM_VERSION=${ROCM_VERSION} --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\"
        elif [ "${DISTRO}" = "opensuse" ]; then
            DISTRO_IMAGE="opensuse/leap"
            if [[ "${VERSION_MAJOR}" -le 15 && "${VERSION_MINOR}" -le 5 ]]; then
                PERL_REPO="15.6"
            else
                PERL_REPO="${VERSION_MAJOR}.${VERSION_MINOR}"
            fi
            verbose-build docker build . ${PULL} --progress plain -f ${DOCKER_FILE} --tag ${CONTAINER} --build-arg DISTRO=${DISTRO_IMAGE} --build-arg VERSION=${VERSION} --build-arg ROCM_VERSION=${ROCM_VERSION} --build-arg PERL_REPO=${PERL_REPO} --build-arg PYTHON_VERSIONS=\"${PYTHON_VERSIONS}\"
        fi
        if [ "${PUSH}" -ne 0 ]; then
            docker push ${CONTAINER}
        fi
    done
done
