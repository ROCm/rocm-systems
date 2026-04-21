#!/bin/bash
#
# Install the AINIC driver inside the container.
#
# Expects the driver source tree to be bind-mounted at AINIC_DRIVER_DIR
# (default: /root/cache/drivers-linux).  Builds rdma-core from the
# vendored source and runs the AINIC setup script.
#
# Idempotent: skips if already installed (marker file in /opt/builds).
#
# Required env for RCCL after installation:
#   NCCL_IB_GID_INDEX=1

set -eo pipefail

AINIC_DRIVER_DIR="${AINIC_DRIVER_DIR:-/root/cache/drivers-linux}"
MARKER="/opt/builds/.ainic-driver.done"
VERBOSE="${VERBOSE:-}"
AINIC_LOG="/tmp/ainic-driver-install.log"

log_verbose() {
    [[ -n "${VERBOSE}" ]] && echo "  [verbose] ainic: $*" || true
}

# When --rebuild is used, clear stale marker so we always reinstall
if [[ "${FORCE_POST_SETUP:-}" == "1" ]] && [[ -f "${MARKER}" ]]; then
    echo "  AINIC: clearing stale marker (FORCE_POST_SETUP=1)"
    rm -f "${MARKER}"
fi

if [[ -f "${MARKER}" ]]; then
    echo "  AINIC: already installed (cached)"
    log_verbose "Marker: ${MARKER}"
    exit 0
fi

if [[ ! -d "${AINIC_DRIVER_DIR}" ]]; then
    echo "  AINIC: driver dir not found at ${AINIC_DRIVER_DIR}, skipping"
    exit 0
fi

if [[ ! -f "${AINIC_DRIVER_DIR}/setup_libs.sh" ]] || [[ ! -d "${AINIC_DRIVER_DIR}/rdma-core" ]]; then
    echo "  AINIC: driver dir ${AINIC_DRIVER_DIR} exists but is missing required files"
    echo "         Expected: setup_libs.sh, rdma-core/"
    echo "         Found:    $(ls "${AINIC_DRIVER_DIR}/" 2>/dev/null || echo '(empty)')"
    echo "         Skipping driver installation."
    exit 0
fi

echo "  AINIC: installing driver from ${AINIC_DRIVER_DIR}"
echo "  AINIC: rdma-core will be installed to /usr (system-wide)"
echo "  AINIC: build log -> ${AINIC_LOG}"

work_dir="/tmp/drivers-linux"
rm -rf "${work_dir}"
cp -r "${AINIC_DRIVER_DIR}" "${work_dir}"

# --- setup_libs.sh ---
echo "  AINIC: [1/3] running setup_libs.sh ..."
cd "${work_dir}"
if [[ -n "${VERBOSE}" ]]; then
    bash ./setup_libs.sh 2>&1 | tee -a "${AINIC_LOG}" | sed 's/^/    [ainic:setup_libs] /'
else
    bash ./setup_libs.sh >> "${AINIC_LOG}" 2>&1
fi

# --- rdma-core build ---
echo "  AINIC: [2/3] building rdma-core ..."
mkdir -p "${work_dir}/rdma-core/build"
cd "${work_dir}/rdma-core/build"
if [[ -n "${VERBOSE}" ]]; then
    cmake -GNinja \
        -DCMAKE_INSTALL_PREFIX:PATH=/usr \
        -DNO_PYVERBS=1 \
        -DNO_MAN_PAGES=1 \
        ${EXTRA_CMAKE_FLAGS:-} .. 2>&1 | tee -a "${AINIC_LOG}" | sed 's/^/    [ainic:cmake] /'
    ninja install 2>&1 | tee -a "${AINIC_LOG}" | sed 's/^/    [ainic:ninja] /'
else
    cmake -GNinja \
        -DCMAKE_INSTALL_PREFIX:PATH=/usr \
        -DNO_PYVERBS=1 \
        -DNO_MAN_PAGES=1 \
        ${EXTRA_CMAKE_FLAGS:-} .. >> "${AINIC_LOG}" 2>&1
    ninja install >> "${AINIC_LOG}" 2>&1
fi

# --- verify ---
echo "  AINIC: [3/3] verifying IB devices ..."
ibv_out=$(ibv_devices 2>&1) || true
echo "${ibv_out}" | sed 's/^/    [ainic:ibv] /'
echo "${ibv_out}" >> "${AINIC_LOG}"

rm -rf "${work_dir}"
touch "${MARKER}" 2>/dev/null || true
echo "  AINIC: driver installed successfully"
log_verbose "Log: ${AINIC_LOG}"
log_verbose "Marker: ${MARKER}"
