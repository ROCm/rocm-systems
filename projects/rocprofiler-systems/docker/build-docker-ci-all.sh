#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

DISTROS=(
    "ubuntu:22.04"
    "ubuntu:24.04"
    "debian:12"
    "rhel:8.10"
    "rhel:9"
    "rhel:10"
)

ROCM_VERSIONS=("6.3" "6.4" "7.0" "7.1" "7.2")

: ${PARALLEL:=$(( $(nproc) / 4 ))}
if [ "${PARALLEL}" -lt 1 ]; then PARALLEL=1; fi
if [ "${PARALLEL}" -gt 8 ]; then PARALLEL=8; fi

LOG_DIR="/tmp/rocprof-sys-docker-builds"

usage() {
    echo "Usage: $(basename $0) [OPTIONS]"
    echo ""
    echo "Build all distro x ROCm version Docker images in parallel."
    echo ""
    echo "Options:"
    echo "  -j, --parallel N     Max concurrent builds (default: ${PARALLEL}, auto-detected from nproc/4)"
    echo "  --log-dir DIR        Directory for build logs (default: ${LOG_DIR})"
    echo "  --distros D1,D2,...  Comma-separated distros to build (default: all)"
    echo "                       Format: distro:version (e.g. ubuntu:22.04,rhel:9)"
    echo "  --rocm V1,V2,...    Comma-separated ROCm versions (default: all)"
    echo "  -h, --help           Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--parallel) shift; PARALLEL="$1"; shift ;;
        --log-dir) shift; LOG_DIR="$1"; shift ;;
        --distros) shift; IFS=',' read -ra DISTROS <<< "$1"; shift ;;
        --rocm) shift; IFS=',' read -ra ROCM_VERSIONS <<< "$1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

TOTAL=$(( ${#DISTROS[@]} * ${#ROCM_VERSIONS[@]} ))
echo -e "${BOLD}Docker CI Image Builder${RESET}"
echo "  Distros:       ${DISTROS[*]}"
echo "  ROCm versions: ${ROCM_VERSIONS[*]}"
echo "  Total images:  ${TOTAL}"
echo "  Parallel:      ${PARALLEL}"
echo "  Log dir:       ${LOG_DIR}"
echo ""

build_one() {
    local distro="$1"
    local version="$2"
    local rocm="$3"
    local log="${LOG_DIR}/build-${distro}-${version}-rocm-${rocm}.log"
    local tag="${distro}-${version}+rocm-${rocm}"

    echo -e "  ${BOLD}START${RESET}  ${tag}  ->  ${log}"

    if "${SCRIPT_DIR}/build-docker-ci.sh" \
        --distro "${distro}" \
        --versions "${version}" \
        --rocm-version "${rocm}" \
        --no-pull \
        > "${log}" 2>&1; then
        echo -e "  ${GREEN}BUILT${RESET}  ${tag}"
        return 0
    else
        echo -e "  ${RED}FAIL ${RESET}  ${tag}  (see ${log})"
        return 1
    fi
}

PIDS=()
TAGS=()
BUILD_PASS=0
BUILD_FAIL=0
BUILD_FAILURES=()

echo -e "${BOLD}=== Building (parallel=${PARALLEL}) ===${RESET}"
echo ""

for rocm in "${ROCM_VERSIONS[@]}"; do
    for entry in "${DISTROS[@]}"; do
        distro="${entry%%:*}"
        version="${entry#*:}"

        while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do
            sleep 2
        done

        build_one "${distro}" "${version}" "${rocm}" &
        PIDS+=($!)
        TAGS+=("${distro}-${version}+rocm-${rocm}")
    done
done

echo ""
echo -e "${BOLD}Waiting for ${#PIDS[@]} builds to finish...${RESET}"

for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        BUILD_PASS=$((BUILD_PASS + 1))
    else
        BUILD_FAIL=$((BUILD_FAIL + 1))
        BUILD_FAILURES+=("${TAGS[$i]}")
    fi
done

echo ""
echo -e "${BOLD}==========================================${RESET}"
echo -e "${BOLD} Results: ${GREEN}${BUILD_PASS} passed${RESET}, ${RED}${BUILD_FAIL} failed${RESET} (out of ${#PIDS[@]})"
echo -e "${BOLD}==========================================${RESET}"

if [ "${BUILD_FAIL}" -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed builds:${RESET}"
    for f in "${BUILD_FAILURES[@]}"; do
        echo -e "  - ${f}"
    done
    echo ""
    echo "Check logs in ${LOG_DIR}/"
    exit 1
fi

echo ""
echo -e "${GREEN}All builds passed.${RESET}"
