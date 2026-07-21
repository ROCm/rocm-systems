#!/usr/bin/env bash
# Download + extract one or more ROCm versions into sibling dirs under
# ${ROCM_ROOT}, the same way the GIN CI does (curl + tar against TheRock's
# flattened dist tarballs; see lib/fetch-rocm.sh). Defaults to durable GA
# (release-channel) builds.
#
# For each requested version we resolve the latest matching release in the
# channel, extract it into ${ROCM_ROOT}/rocm-<full-release>, and refresh a
# convenience symlink ${ROCM_ROOT}/rocm-<requested> -> rocm-<full-release>.
#
# Environment overrides:
#   ROCM_ROOT             Install root (default: /shared_inference/rocm)
#   ROCM_VERSIONS         Space-separated versions to fetch, e.g. "7.13 7.14"
#                         or "7.13.0" (required; no default so the script never
#                         goes stale across ROCm releases).
#   ROCM_RELEASE_CHANNEL  release|prerelease|nightly|dev (default: release)
#   ROCM_AMDGPU_FAMILY    Artifact family (default: gfx950-dcgpu)
#   THEROCK_TARBALL_BASE_URL  Override the per-channel base URL

set -euo pipefail

ROCM_ROOT="${ROCM_ROOT:-/shared_inference/rocm}"
rocm_family="${ROCM_AMDGPU_FAMILY:-gfx950-dcgpu}"
rocm_channel="${ROCM_RELEASE_CHANNEL:-release}"

if [[ -z "${ROCM_VERSIONS:-}" ]]; then
  echo "ERROR: set ROCM_VERSIONS to the versions to fetch, e.g. ROCM_VERSIONS='7.13 7.14'" >&2
  exit 1
fi
read -r -a versions <<< "${ROCM_VERSIONS}"

case "${rocm_channel}" in
  nightly)    default_base_url="https://therock-nightly-tarball.s3.amazonaws.com"; list_style="s3"  ;;
  dev)        default_base_url="https://therock-dev-tarball.s3.amazonaws.com";     list_style="s3"  ;;
  prerelease) default_base_url="https://rocm.prereleases.amd.com/tarball";         list_style="cdn" ;;
  release)    default_base_url="https://repo.amd.com/rocm/tarball";                list_style="cdn" ;;
  *)
    echo "ERROR: unknown ROCM_RELEASE_CHANNEL='${rocm_channel}'" >&2
    exit 1 ;;
esac
base_url="${THEROCK_TARBALL_BASE_URL:-${default_base_url}}"

mkdir -p "${ROCM_ROOT}/downloads"

resolve_latest() {
  # Echo the latest full release string matching a given version prefix.
  local req="$1"
  local prefix="therock-dist-linux-${rocm_family}-${req}"
  local list_url listing matches
  if [[ "${list_style}" == "s3" ]]; then
    list_url="${base_url}/?list-type=2&prefix=${prefix}"
  else
    list_url="${base_url}/"
  fi
  listing="$(curl -fsSL "${list_url}")" || {
    echo "ERROR: failed to list ${list_url}" >&2
    return 1
  }
  matches="$(printf '%s' "${listing}" \
    | grep -oE "${prefix}[0-9.]*[a-z0-9]*\.tar\.gz" \
    | sed -E "s/^therock-dist-linux-${rocm_family}-(.+)\.tar\.gz$/\1/")"
  if [[ "${rocm_channel}" == "release" ]]; then
    matches="$(printf '%s\n' "${matches}" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' || true)"
  fi
  printf '%s\n' "${matches}" | sort -V | tail -n1
}

fetch_one() {
  local req="$1" release rocm_path tarball url dl

  echo "==> Resolving latest ${rocm_channel} for ROCm ${req} (${rocm_family})"
  release="$(resolve_latest "${req}")"
  if [[ -z "${release}" ]]; then
    echo "ERROR: could not find any ${req}* tarball at ${base_url}" >&2
    return 1
  fi
  echo "    latest: ${release}"

  rocm_path="${ROCM_ROOT}/rocm-${release}"
  tarball="therock-dist-linux-${rocm_family}-${release}.tar.gz"
  url="${base_url}/${tarball}"

  if [[ -x "${rocm_path}/bin/hipcc" ]] \
     && grep -qx "family=${rocm_family}" "${rocm_path}/.stamp" 2>/dev/null; then
    echo "    reusing cached install at ${rocm_path}"
  else
    echo "    fetching from ${url}"
    dl="${ROCM_ROOT}/downloads/${tarball}.part"
    rm -f "${dl}"
    curl -fL --retry 5 --retry-delay 5 --retry-connrefused -o "${dl}" "${url}"
    rm -rf "${rocm_path}"
    mkdir -p "${rocm_path}"
    tar -xzf "${dl}" -C "${rocm_path}"
    rm -f "${dl}"
    if [[ ! -x "${rocm_path}/bin/hipcc" ]]; then
      echo "ERROR: extracted ROCm at ${rocm_path} has no bin/hipcc" >&2
      ls -la "${rocm_path}" >&2 || true
      return 1
    fi
    printf 'release=%s\nfamily=%s\n' "${release}" "${rocm_family}" > "${rocm_path}/.stamp"
  fi

  ln -sfn "rocm-${release}" "${ROCM_ROOT}/rocm-${req}"
  echo "    symlink: ${ROCM_ROOT}/rocm-${req} -> rocm-${release}"
}

for req in "${versions[@]}"; do
  fetch_one "${req}"
done

echo
echo "==> Done. ROCm trees under ${ROCM_ROOT}:"
for req in "${versions[@]}"; do
  printf '    %-12s -> %s\n' "rocm-${req}" "$(readlink "${ROCM_ROOT}/rocm-${req}" 2>/dev/null || echo '<missing>')"
done
