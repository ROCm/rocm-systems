#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build script for NIC transport layer tests.
#
# The static library location is resolved in this order:
#   1. first positional argument
#   2. $AMDSMI_BUILD_DIR
#   3. a search of the enclosing workspace for libamdsminic.a
# Test binaries are written under $TMPDIR (default /tmp).

set -Eeuo pipefail
shopt -s inherit_errexit

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly PROJECT_ROOT
readonly INC_DIR="${PROJECT_ROOT}/src/nic/ai-nic/amdsmi_unified/inc"
readonly LIB_NAME="libamdsminic.a"
readonly OUT_DIR="${TMPDIR:-/tmp}"

# Test sources (relative to SCRIPT_DIR, without extension) built by this script.
readonly TEST_TARGETS=(
  test_nic_transport_suite
  benchmark_nic_transport
  test_nic_telemetry_suite
  test_nic_vpd_suite
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
  [[ -d "${INC_DIR}" ]] || die "include directory not found: ${INC_DIR}"

  # Resolve the static library location. When several build trees exist, pick the
  # most recently built archive so a stale copy is never chosen silently.
  local build_dir="${1:-${AMDSMI_BUILD_DIR:-}}"
  local search_root="${build_dir}"
  if [[ -z "${search_root}" ]]; then
    search_root=$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)
  fi

  local -a candidates=()
  mapfile -t -d '' candidates < <(find "${search_root}" -type f -name "${LIB_NAME}" -print0 2>/dev/null || true)
  if [[ ${#candidates[@]} -eq 0 ]]; then
    warn "Could not find ${LIB_NAME} under ${search_root}."
    warn "Build it first (e.g. 'make amdsminic' in your CMake build dir), then pass"
    warn "the build directory as the first argument or via \$AMDSMI_BUILD_DIR."
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

  # Optional netlink support via libnl-3.
  local -a libnl_cflags=() libnl_libs=() have_libnl3=()
  if command -v pkg-config &>/dev/null &&
    pkg-config --exists libnl-3.0 libnl-genl-3.0 2>/dev/null; then
    local cflags_str libs_str
    cflags_str=$(pkg-config --cflags libnl-3.0 libnl-genl-3.0)
    libs_str=$(pkg-config --libs libnl-3.0 libnl-genl-3.0)
    read -r -a libnl_cflags <<<"${cflags_str}"
    read -r -a libnl_libs <<<"${libs_str}"
    have_libnl3=(-DHAVE_LIBNL3)
    log "Netlink support: YES (libnl-3 detected)"
  else
    log "Netlink support: NO (libnl-3 not found)"
  fi

  # -fno-rtti matches the ABI of libamdsminic.a (built without RTTI). A test that
  # subclasses a library type (e.g. SmiNic) otherwise fails to link with an
  # undefined typeinfo reference.
  local -a cxxflags=(
    -std=c++17 -Wall -Wextra -Werror -fno-rtti
    -I "${INC_DIR}"
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
  log "  # telemetry suite is self-contained (fakes + tmpdir, no interface/root):"
  log "  ${OUT_DIR}/test_nic_telemetry_suite"
  log "  # live devlink probes (need a devlink NIC + root):"
  log "  sudo ${OUT_DIR}/probe_devlink_health <pci-bdf>   # e.g. 0000:63:00.0"
  log "  sudo ${OUT_DIR}/probe_devlink_port   <pci-bdf>   # e.g. 0000:63:00.0"
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
