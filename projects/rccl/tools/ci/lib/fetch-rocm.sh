#!/usr/bin/env bash
# Resolve + download a ROCm dist tarball from TheRock and write .ci-out/rocm.env
# (ROCM_PATH, ROCM_RELEASE) for later stages. Bash (not
# install_rocm_from_artifacts.py) because the head node only has Python 3.9.
#
# Resolves the "best" tarball for a requested version via a stability-first
# cascade over three public tarball-multi-arch CDN indexes (no token needed):
#
#   1. release     repo.amd.com/rocm/tarball-multi-arch        (stable X.Y.Z)
#   2. prerelease  rocm.prereleases.amd.com/tarball-multi-arch (rc builds)
#   3. nightly     rocm.nightlies.amd.com/tarball-multi-arch   (dated alphas)
#
# Each is listed the same way (GET index, scrape names, pick greatest match) and
# downloaded with a plain curl. The FIRST (channel, family) match wins, so a
# release is preferred over a newer nightly ("greatest, not latest") and the
# multi-arch tarball over a single-family one.
#
# Tarball naming: therock-dist-linux-<family>-<version>.tar.gz
# (<family> e.g. "multiarch" or "gfx950-dcgpu").
#
# Environment:
#   ROCM_RELEASE              Version query, matched as a prefix: "7.14" picks the
#                            newest 7.14.*; "7.14.0rc0" pins exactly; empty => newest.
#   ROCM_AMDGPU_FAMILY        Specific GPU family fallback (default: gfx950-dcgpu)
#   RCCL_ROCM_CHANNELS        Space-separated cascade override
#                            (default: "release prerelease nightly")
#   RCCL_ROCM_TRY_MULTIARCH   1 (default) to try the multiarch tarball first; 0 to skip
#   RCCL_DEVICE_API_CACHE     Cache root (default: /apps/rccl-ci)
#   ROCM_PATH_OVERRIDE        Use this existing ROCm tree; skips resolve+download
#   ROCM_REF_TOKEN            If set, register a reference for GC (see rocm-ref.sh)
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)
#   RCCL_ROCM_RESOLVE_ONLY    1 => print resolved channel/family/version/url, no download
#   RCCL_CI_TRACE             1 => enable `set -x` tracing (off by default)
#
# Per-channel base-URL overrides (rarely needed): RELEASE_TARBALL_URL,
# PRERELEASE_TARBALL_URL, NIGHTLY_TARBALL_URL, DEV_TARBALL_URL
#
# Legacy: ROCM_RELEASE_CHANNEL (release|prerelease|nightly|dev), when set (and
# RCCL_ROCM_CHANNELS is not), pins selection to that single channel.

set -euo pipefail
if [[ -n "${RCCL_CI_TRACE:-}" ]]; then set -x; fi

VERSION_QUERY="${ROCM_RELEASE:-}"
SPECIFIC_FAMILY="${ROCM_AMDGPU_FAMILY:-gfx950-dcgpu}"

RELEASE_TARBALL_URL="${RELEASE_TARBALL_URL:-https://repo.amd.com/rocm/tarball-multi-arch}"
PRERELEASE_TARBALL_URL="${PRERELEASE_TARBALL_URL:-https://rocm.prereleases.amd.com/tarball-multi-arch}"
NIGHTLY_TARBALL_URL="${NIGHTLY_TARBALL_URL:-https://rocm.nightlies.amd.com/tarball-multi-arch}"
DEV_TARBALL_URL="${DEV_TARBALL_URL:-https://rocm.devreleases.amd.com/tarball-multi-arch}"

# Channel selection: an explicit RCCL_ROCM_CHANNELS wins; else the legacy single
# ROCM_RELEASE_CHANNEL; else the default stability-first cascade.
if [[ -n "${RCCL_ROCM_CHANNELS:-}" ]]; then
  read -ra CHANNELS <<< "${RCCL_ROCM_CHANNELS}"
elif [[ -n "${ROCM_RELEASE_CHANNEL:-}" ]]; then
  read -ra CHANNELS <<< "${ROCM_RELEASE_CHANNEL}"
else
  read -ra CHANNELS <<< "release prerelease nightly"
fi

# Family preference: multiarch first (combined, all-arch), then the specific one.
FAMILIES=()
if [[ "${RCCL_ROCM_TRY_MULTIARCH:-1}" == "1" && "${SPECIFIC_FAMILY}" != "multiarch" ]]; then
  FAMILIES+=("multiarch")
fi
FAMILIES+=("${SPECIFIC_FAMILY}")

WORKDIR="${RCCL_DEVICE_API_WORKDIR:-${WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
fi

CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"
env_out="${WORKDIR}/.ci-out/rocm.env"

# --- Helpers ----------------------------------------------------------------

# Print the newest tarball <version> for a family from a CDN index page on stdin,
# honoring VERSION_QUERY (prefix) and an optional stable-only (X.Y.Z) filter.
#   $1 = family   $2 = stable_only(0|1)
_pick_latest_version() {
  local family="$1" stable_only="$2"
  local prefix="therock-dist-linux-${family}-"
  local esc="${VERSION_QUERY//./\\.}"
  # [^"<]* stops at an HTML quote or an XML tag boundary.
  grep -oE "${prefix}[0-9][^\"<]*\.tar\.gz" 2>/dev/null \
    | grep -v -- '-tests-' \
    | sed -E "s/^${prefix}(.+)\.tar\.gz$/\1/" \
    | { if [[ -n "${esc}" ]]; then grep -E "^${esc}([.a-z-]|$)" || true; else cat; fi } \
    | { if [[ "${stable_only}" == "1" ]]; then grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' || true; else cat; fi } \
    | sort -V | tail -n1
}

# Resolve against a flat listing endpoint (a tarball-multi-arch CDN index page).
#   $1 = base_url  $2 = stable_only  $3 = family  $4 = list-suffix (usually "/")
# Prints "<version>\t<download_url>" or nothing.
_resolve_flat() {
  local base_url="$1" stable_only="$2" family="$3" list_suffix="$4"
  local listing ver
  listing="$(curl -fsSL "${base_url}${list_suffix}" 2>/dev/null || true)"
  [[ -n "${listing}" ]] || return 0
  ver="$(printf '%s' "${listing}" | _pick_latest_version "${family}" "${stable_only}")"
  [[ -n "${ver}" ]] || return 0
  printf '%s\t%s\n' "${ver}" "${base_url}/therock-dist-linux-${family}-${ver}.tar.gz"
}

# Dispatch one (channel, family) probe. Prints "<version>\t<url>" or nothing.
_resolve_one() {
  local channel="$1" family="$2"
  case "${channel}" in
    release)    _resolve_flat "${RELEASE_TARBALL_URL}"    1 "${family}" "/" ;;
    prerelease) _resolve_flat "${PRERELEASE_TARBALL_URL}" 0 "${family}" "/" ;;
    nightly)    _resolve_flat "${NIGHTLY_TARBALL_URL}"    0 "${family}" "/" ;;
    dev)        _resolve_flat "${DEV_TARBALL_URL}"        0 "${family}" "/" ;;
    *) echo "ERROR: unknown channel '${channel}' in RCCL_ROCM_CHANNELS" >&2; return 1 ;;
  esac
}

# --- ROCM_PATH_OVERRIDE: explicit tree, no resolve/download -----------------

mkdir -p "${WORKDIR}/.ci-out"
if [[ -n "${ROCM_PATH_OVERRIDE:-}" ]]; then
  ROCM_PATH="${ROCM_PATH_OVERRIDE}"
  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: ROCM_PATH_OVERRIDE=${ROCM_PATH} has no bin/hipcc" >&2
    exit 1
  fi
  echo "==> Using ROCM_PATH_OVERRIDE=${ROCM_PATH} (no fetch)"
  {
    printf 'export ROCM_RELEASE=%q\n' "${ROCM_RELEASE:-}"
    printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}"
  } > "${env_out}"
  exit 0
fi

# --- Resolve via the cascade ------------------------------------------------

echo "==> Resolving ROCm: version-query='${VERSION_QUERY:-<latest>}' families='${FAMILIES[*]}'"
echo "==> Cascade: ${CHANNELS[*]}"

WIN_VER="" WIN_URL="" WIN_CHANNEL="" WIN_FAMILY=""
for channel in "${CHANNELS[@]}"; do
  for family in "${FAMILIES[@]}"; do
    res="$(_resolve_one "${channel}" "${family}" || true)"
    if [[ -n "${res}" ]]; then
      WIN_VER="${res%%$'\t'*}"
      WIN_URL="${res#*$'\t'}"
      WIN_CHANNEL="${channel}"
      WIN_FAMILY="${family}"
      echo "==> MATCH: channel=${channel} family=${family} version=${WIN_VER}"
      break 2
    fi
    echo "    miss: channel=${channel} family=${family}"
  done
done

if [[ -z "${WIN_URL}" ]]; then
  echo "ERROR: no ROCm tarball found for version-query='${VERSION_QUERY:-<latest>}'" >&2
  echo "       across channels [${CHANNELS[*]}] and families [${FAMILIES[*]}]" >&2
  exit 1
fi

echo "==> Selected: ${WIN_CHANNEL}/${WIN_FAMILY} ROCm ${WIN_VER}"
echo "==>       url: ${WIN_URL}"

if [[ -n "${RCCL_ROCM_RESOLVE_ONLY:-}" ]]; then
  echo "==> RCCL_ROCM_RESOLVE_ONLY set; not downloading."
  exit 0
fi

# --- Download + cache -------------------------------------------------------

if ! mkdir -p "${CACHE_DIR}/rocm" "${CACHE_DIR}/downloads"; then
  echo "ERROR: cannot create cache dirs under ${CACHE_DIR} (is /apps writable?)" >&2
  exit 1
fi

ROCM_RELEASE="${WIN_VER}"
# Version-only cache path (must match rocm-ref.sh for refcounting/GC). The
# resolved family/channel are recorded in .stamp, not the path.
ROCM_PATH="${CACHE_DIR}/rocm/rocm-${ROCM_RELEASE}"
refs_dir="${ROCM_PATH}.refs"
ref_token="${ROCM_REF_TOKEN:-}"
tarball="$(basename "${WIN_URL}")"

cache_hit() {
  [[ -x "${ROCM_PATH}/bin/hipcc" ]] \
    && grep -qx "family=${WIN_FAMILY}" "${ROCM_PATH}/.stamp" 2>/dev/null \
    && grep -qx "channel=${WIN_CHANNEL}" "${ROCM_PATH}/.stamp" 2>/dev/null \
    && grep -qx "release=${ROCM_RELEASE}" "${ROCM_PATH}/.stamp" 2>/dev/null
}

do_fetch() {
  echo "==> Fetching ROCm ${ROCM_RELEASE} (${WIN_CHANNEL}/${WIN_FAMILY}) from ${WIN_URL}"
  local dl="${CACHE_DIR}/downloads/${tarball}.$$.part"
  rm -f "${dl}"
  # -f makes HTTP errors fail (instead of saving an XML/HTML error body as the tarball).
  curl -fL --retry 5 --retry-delay 5 --retry-connrefused -o "${dl}" "${WIN_URL}"
  rm -rf "${ROCM_PATH}"
  mkdir -p "${ROCM_PATH}"
  tar -xzf "${dl}" -C "${ROCM_PATH}"
  rm -f "${dl}"
  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: extracted ROCm at ${ROCM_PATH} has no bin/hipcc" >&2
    ls -la "${ROCM_PATH}" >&2 || true
    exit 1
  fi
  printf 'release=%s\nfamily=%s\nchannel=%s\n' \
    "${ROCM_RELEASE}" "${WIN_FAMILY}" "${WIN_CHANNEL}" > "${ROCM_PATH}/.stamp"
}

acquire_ref() {
  [[ -n "${ref_token}" ]] || return 0
  mkdir -p "${refs_dir}"
  : > "${refs_dir}/${ref_token}"
  echo "==> Registered ROCm reference ${ref_token} (channel=${WIN_CHANNEL})"
}

# Per-version lock serializes parallel jobs (the loser reuses the winner's tree);
# with a ref token we always lock, even on a cache hit, to order against GC.
if cache_hit && [[ -z "${ref_token}" ]]; then
  echo "==> Reusing cached ROCm ${ROCM_RELEASE} at ${ROCM_PATH}"
else
  exec {lock_fd}>"${CACHE_DIR}/downloads/.lock-${ROCM_RELEASE}"
  flock "${lock_fd}"
  acquire_ref
  if cache_hit; then
    echo "==> Reusing cached ROCm ${ROCM_RELEASE} at ${ROCM_PATH}"
  else
    do_fetch
  fi
  flock -u "${lock_fd}"
fi

{
  printf 'export ROCM_RELEASE=%q\n' "${ROCM_RELEASE}"
  printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}"
} > "${env_out}"
echo "==> Wrote ${env_out}"
