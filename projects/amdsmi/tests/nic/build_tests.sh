#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Build script for NIC transport layer tests.
#
# The CMake build directory holding libamdsminic.a is the first positional argument or
# $AMDSMI_BUILD_DIR; there is no workspace-wide search, which could pair these sources with an
# archive from another build. Netlink support follows how that archive was built.
# Test binaries are written under $TMPDIR (default /tmp).

set -Eeuo pipefail
shopt -s inherit_errexit

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly PROJECT_ROOT
readonly INC_DIR="${PROJECT_ROOT}/src/nic/ai-nic/amdsmi_unified/inc"
# Vendor plugin headers are included as "vendors/<name>/...", relative to src/.
readonly SRC_DIR="${PROJECT_ROOT}/src/nic/ai-nic/amdsmi_unified/src"
readonly IFACE_DIR="${PROJECT_ROOT}/include/amd_smi/impl/nic/amdsmi_unified/interface"
readonly LIB_NAME="libamdsminic.a"
readonly OUT_DIR="${TMPDIR:-/tmp}"

# Test sources (relative to SCRIPT_DIR, without extension) built by this script.
readonly TEST_TARGETS=(
  test_nic_transport_suite
  benchmark_nic_transport
  test_nic_telemetry_suite
  test_nic_vpd_suite
  test_nic_log_suite
  test_nic_discovery_suite
  test_nic_stats_suite
  test_nic_check_suite
  probe_devlink_health
  probe_devlink_port
)

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require() {
  command -v "$1" &>/dev/null || die "required tool not found: $1"
}

main() {
  require g++
  require find
  require ar
  [[ -d "${INC_DIR}" ]] || die "include directory not found: ${INC_DIR}"

  # Resolve the static library location. When the build directory holds several copies,
  # pick the most recently built so a stale one is never chosen silently.
  local search_root="${1:-${AMDSMI_BUILD_DIR:-}}"
  if [[ -z "${search_root}" ]]; then
    warn "No build directory given. Build the library first (e.g. 'make amdsminic' in your"
    warn "CMake build dir), then pass that directory as the first argument or via"
    warn "\$AMDSMI_BUILD_DIR."
    die "build directory required"
  fi

  local -a candidates=()
  mapfile -t -d '' candidates < <(find "${search_root}" -type f -name "${LIB_NAME}" -print0 2>/dev/null || true)
  if [[ ${#candidates[@]} -eq 0 ]]; then
    warn "Could not find ${LIB_NAME} under ${search_root}."
    warn "Build it first (e.g. 'make amdsminic' in your CMake build dir)."
    die "${LIB_NAME} not found"
  fi

  local lib_path="" newest=0 candidate mtime
  for candidate in "${candidates[@]}"; do
    mtime=$(stat -c '%Y' "${candidate}")
    if ((mtime > newest)); then
      newest="${mtime}"
      lib_path="${candidate}"
    fi
  done
  if ((${#candidates[@]} > 1)); then
    warn "Found ${#candidates[@]} copies of ${LIB_NAME}; using the newest: ${lib_path}"
  fi

  log "+-------------------------------------------------------------------+"
  log "Building NIC Transport Layer Tests"
  log "+-------------------------------------------------------------------+"
  log "Project root: ${PROJECT_ROOT}"
  log "Library:      ${lib_path}"
  log "Output dir:   ${OUT_DIR}"

  # Netlink support follows the archive: it carries the netlink transport only when it was built
  # with libnl-3, and the tests must be compiled and linked the same way.
  local -a libnl_cflags=() libnl_libs=() have_libnl3=()
  if ar t "${lib_path}" | grep -q '^smi_nic_transport_netlink\.cpp\.o$'; then
    if ! command -v pkg-config &>/dev/null || ! pkg-config --exists libnl-3.0 libnl-genl-3.0 2>/dev/null; then
      die "${lib_path} was built with netlink but libnl-3 was not found via pkg-config"
    fi
    local cflags_str libs_str
    cflags_str=$(pkg-config --cflags libnl-3.0 libnl-genl-3.0)
    libs_str=$(pkg-config --libs libnl-3.0 libnl-genl-3.0)
    read -r -a libnl_cflags <<<"${cflags_str}"
    read -r -a libnl_libs <<<"${libs_str}"
    have_libnl3=(-DHAVE_LIBNL3)
    log "Netlink support: YES (the library was built with it)"
  else
    log "Netlink support: NO (the library was built without it)"
  fi

  # -fno-rtti matches the ABI of libamdsminic.a (built without RTTI). A test that
  # subclasses a library type (e.g. SmiNic) otherwise fails to link with an
  # undefined typeinfo reference.
  local -a cxxflags=(
    -std=c++17 -Wall -Wextra -Werror -fno-rtti
    -I "${INC_DIR}" -I "${SRC_DIR}" -I "${IFACE_DIR}"
    # The header-only suites include amd_smi/... from the public include root. -isystem keeps
    # warnings in the public header (amdsmi.h) from tripping -Werror.
    -isystem "${PROJECT_ROOT}/include"
    "${have_libnl3[@]}" "${libnl_cflags[@]}"
  )

  local target src out
  for target in "${TEST_TARGETS[@]}"; do
    src="${SCRIPT_DIR}/${target}.cpp"
    out="${OUT_DIR}/${target}"
    [[ -f "${src}" ]] || die "test source not found: ${src}"
    log "Building: ${target} ..."
    g++ "${cxxflags[@]}" -o "${out}" "${src}" "${lib_path}" "${libnl_libs[@]}"
    log "  -> ${out}"
  done

  log "+-------------------------------------------------------------------+"
  log "Build complete. Run:"
  log "  # transport/benchmark need a live interface and root:"
  log "  sudo ${OUT_DIR}/test_nic_transport_suite <interface>"
  log "  sudo ${OUT_DIR}/benchmark_nic_transport <interface> [iterations]"
  log "  # fakes + tmpdir, no interface/root (telemetry also reads the host /sys):"
  log "  ${OUT_DIR}/test_nic_telemetry_suite"
  log "  ${OUT_DIR}/test_nic_discovery_suite"
  log "  ${OUT_DIR}/test_nic_vpd_suite"
  log "  ${OUT_DIR}/test_nic_log_suite"
  log "  ${OUT_DIR}/test_nic_stats_suite"
  log "  ${OUT_DIR}/test_nic_check_suite"
  log "  # live devlink probes (need a devlink NIC + root):"
  log "  sudo ${OUT_DIR}/probe_devlink_health <pci-bdf>   # e.g. 0000:63:00.0"
  log "  sudo ${OUT_DIR}/probe_devlink_port   <pci-bdf>   # e.g. 0000:63:00.0"
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
