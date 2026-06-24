#!/usr/bin/env bash
# Download + extract a ROCm dist tarball (curl + tar) from a TheRock channel and
# write .ci-out/rocm.env (ROCM_PATH, ROCM_RELEASE) for later stages. Bash (not
# install_rocm_from_artifacts.py) because the head node only has Python 3.9.
#
# Environment:
#   RCCL_DEVICE_API_CACHE     Cache root (default: /apps/rccl-ci)
#   ROCM_RELEASE_CHANNEL      nightly|dev|prerelease|release (default: nightly)
#   ROCM_RELEASE              Version, empty => latest for the channel
#   ROCM_AMDGPU_FAMILY        Artifact family (default: gfx950-dcgpu)
#   ROCM_PATH_OVERRIDE        Use this existing ROCm tree; skips the download
#   THEROCK_TARBALL_BASE_URL  Override the per-channel base URL
#   ROCM_REF_TOKEN            If set, register a reference for GC (see rocm-ref.sh)
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)

set -euxo pipefail

rocm_family="${ROCM_AMDGPU_FAMILY:-gfx950-dcgpu}"
rocm_channel="${ROCM_RELEASE_CHANNEL:-nightly}"

# Per-channel base URL (overridable via THEROCK_TARBALL_BASE_URL). list_style
# picks how "latest" is resolved: S3 XML (nightly/dev) vs HTML index (CDNs).
case "${rocm_channel}" in
  nightly)    default_base_url="https://therock-nightly-tarball.s3.amazonaws.com"; list_style="s3"  ;;
  dev)        default_base_url="https://therock-dev-tarball.s3.amazonaws.com";     list_style="s3"  ;;
  prerelease) default_base_url="https://rocm.prereleases.amd.com/tarball";         list_style="cdn" ;;
  release)    default_base_url="https://repo.amd.com/rocm/tarball";                list_style="cdn" ;;
  *)
    echo "ERROR: unknown ROCM_RELEASE_CHANNEL='${rocm_channel}'" >&2
    echo "       expected one of: nightly, dev, prerelease, release" >&2
    exit 1 ;;
esac
base_url="${THEROCK_TARBALL_BASE_URL:-${default_base_url}}"

WORKDIR="${RCCL_DEVICE_API_WORKDIR:-${WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
fi

CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"
if ! mkdir -p "${CACHE_DIR}/rocm" "${CACHE_DIR}/downloads"; then
  echo "ERROR: cannot create cache dirs under ${CACHE_DIR} (is /apps writable?)" >&2
  exit 1
fi

env_out="${WORKDIR}/.ci-out/rocm.env"
mkdir -p "${WORKDIR}/.ci-out"

# Explicit override only (ambient ROCM_PATH is never trusted).
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

unset ROCM_PATH

# Resolve a concrete version (empty ROCM_RELEASE => latest for the channel).
if [[ -z "${ROCM_RELEASE:-}" ]]; then
  echo "==> ROCM_RELEASE unset; resolving latest '${rocm_channel}' for ${rocm_family}"
  prefix="therock-dist-linux-${rocm_family}-"
  if [[ "${list_style}" == "s3" ]]; then
    list_url="${base_url}/?list-type=2&prefix=${prefix}"
  else
    list_url="${base_url}/"
  fi
  listing="$(curl -fsSL "${list_url}")" || {
    echo "ERROR: failed to list ${list_url}" >&2
    exit 1
  }
  # Version strings from the tarball names ([^"<]* stops at HTML quote / XML tag).
  versions="$(printf '%s' "${listing}" \
    | grep -oE "${prefix}[0-9][^\"<]*\.tar\.gz" \
    | sed -E "s/^${prefix}(.+)\.tar\.gz$/\1/")"
  # Stable channel: keep only X.Y.Z (drop rc/alpha).
  if [[ "${rocm_channel}" == "release" ]]; then
    versions="$(printf '%s\n' "${versions}" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' || true)"
  fi
  ROCM_RELEASE="$(printf '%s\n' "${versions}" | sort -V | tail -n1 || true)"
  if [[ -z "${ROCM_RELEASE}" ]]; then
    echo "ERROR: could not find any ${prefix}*.tar.gz at ${base_url}" >&2
    exit 1
  fi
  echo "==> Latest ${rocm_channel}: ${ROCM_RELEASE}"
fi

ROCM_PATH="${CACHE_DIR}/rocm/rocm-${ROCM_RELEASE}"
refs_dir="${ROCM_PATH}.refs"
ref_token="${ROCM_REF_TOKEN:-}"
tarball="therock-dist-linux-${rocm_family}-${ROCM_RELEASE}.tar.gz"
url="${base_url}/${tarball}"

cache_hit() {
  [[ -x "${ROCM_PATH}/bin/hipcc" ]] \
    && grep -qx "family=${rocm_family}" "${ROCM_PATH}/.stamp" 2>/dev/null \
    && grep -qx "channel=${rocm_channel}" "${ROCM_PATH}/.stamp" 2>/dev/null
}

do_fetch() {
  echo "==> Fetching ROCm ${ROCM_RELEASE} (${rocm_family}) from ${url}"
  # Unique .part so concurrent fetchers never share a temp file.
  dl="${CACHE_DIR}/downloads/${tarball}.$$.part"
  rm -f "${dl}"
  # -f makes HTTP errors fail (instead of saving an XML error body as the tarball).
  curl -fL --retry 5 --retry-delay 5 --retry-connrefused -o "${dl}" "${url}"
  # The dist tarball is flattened (./bin, ./lib, ...); extract into the
  # version-keyed dir to get ${ROCM_PATH}/bin/hipcc.
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
    "${ROCM_RELEASE}" "${rocm_family}" "${rocm_channel}" > "${ROCM_PATH}/.stamp"
}

# Register before the cache decision so another run's GC can't delete the tree
# between our check and our use.
acquire_ref() {
  [[ -n "${ref_token}" ]] || return 0
  mkdir -p "${refs_dir}"
  : > "${refs_dir}/${ref_token}"
  echo "==> Registered ROCm reference ${ref_token} (channel=${rocm_channel})"
}

# A per-version lock serializes parallel jobs; the inner re-check lets the loser
# reuse the winner's tree. With a ref token we always lock (even on a cache hit)
# so registration is ordered against GC.
if cache_hit && [[ -z "${ref_token}" ]]; then
  echo "==> Reusing cached ROCm ${ROCM_RELEASE} at ${ROCM_PATH}"
else
  exec {lock_fd}>"${CACHE_DIR}/downloads/.lock-${rocm_family}-${rocm_channel}-${ROCM_RELEASE}"
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
