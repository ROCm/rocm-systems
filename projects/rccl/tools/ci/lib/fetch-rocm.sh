#!/usr/bin/env bash
# Download + extract a ROCm tree from TheRock's flattened nightly dist tarball
# (curl + tar). We avoid install_rocm_from_artifacts.py because it needs Python
# >= 3.10, which the cluster only provides inside conda envs. Writes
# .ci-out/rocm.env (ROCM_PATH, ROCM_RELEASE) for the later build/run stages.
#
# Environment:
#   RCCL_DEVICE_API_CACHE     Persistent cache root (default: /apps/rccl-ci)
#   ROCM_RELEASE              Version, e.g. 7.13.0a20260515 (empty => latest)
#   ROCM_AMDGPU_FAMILY        Artifact family (default: gfx950-dcgpu)
#   ROCM_PATH_OVERRIDE        Use this existing ROCm tree; skips the download.
#                             Ambient ROCM_PATH is ignored on purpose, so a stray
#                             `module load rocm` can't substitute /opt/rocm.
#   THEROCK_TARBALL_BASE_URL  Tarball base URL (default: the nightly S3 bucket)
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)

set -euxo pipefail

rocm_family="${ROCM_AMDGPU_FAMILY:-gfx950-dcgpu}"
base_url="${THEROCK_TARBALL_BASE_URL:-https://therock-nightly-tarball.s3.amazonaws.com}"

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

# Resolve a concrete version (empty ROCM_RELEASE => latest in the bucket).
if [[ -z "${ROCM_RELEASE:-}" ]]; then
  echo "==> ROCM_RELEASE unset; resolving latest nightly for ${rocm_family}"
  prefix="therock-dist-linux-${rocm_family}-"
  listing="$(curl -fsSL "${base_url}/?list-type=2&prefix=${prefix}")" || {
    echo "ERROR: failed to list ${base_url} for ${prefix}*" >&2
    exit 1
  }
  ROCM_RELEASE="$(printf '%s' "${listing}" \
    | grep -oE "${prefix}[0-9][^<]*\.tar\.gz" \
    | sed -E "s/^${prefix}(.+)\.tar\.gz$/\1/" \
    | sort -V | tail -n1 || true)"
  if [[ -z "${ROCM_RELEASE}" ]]; then
    echo "ERROR: could not find any ${prefix}*.tar.gz in the bucket listing" >&2
    exit 1
  fi
  echo "==> Latest nightly: ${ROCM_RELEASE}"
fi

ROCM_PATH="${CACHE_DIR}/rocm/rocm-${ROCM_RELEASE}"
tarball="therock-dist-linux-${rocm_family}-${ROCM_RELEASE}.tar.gz"
url="${base_url}/${tarball}"

# Download on cache miss only.
if [[ -x "${ROCM_PATH}/bin/hipcc" ]] \
   && grep -qx "family=${rocm_family}" "${ROCM_PATH}/.stamp" 2>/dev/null; then
  echo "==> Reusing cached ROCm ${ROCM_RELEASE} at ${ROCM_PATH}"
else
  echo "==> Fetching ROCm ${ROCM_RELEASE} (${rocm_family})"
  echo "    from ${url}"
  dl="${CACHE_DIR}/downloads/${tarball}.part"
  rm -f "${dl}"
  # -f makes HTTP errors fail (instead of saving an XML error body as the tarball).
  curl -fL --retry 5 --retry-delay 5 --retry-connrefused \
       -o "${dl}" "${url}"

  # The dist tarball is flattened (./bin, ./lib, ...), so extract straight into
  # the version-keyed dir to get ${ROCM_PATH}/bin/hipcc.
  rm -rf "${ROCM_PATH}"
  mkdir -p "${ROCM_PATH}"
  tar -xzf "${dl}" -C "${ROCM_PATH}"
  rm -f "${dl}"

  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: extracted ROCm at ${ROCM_PATH} has no bin/hipcc" >&2
    ls -la "${ROCM_PATH}" >&2 || true
    exit 1
  fi
  printf 'release=%s\nfamily=%s\n' "${ROCM_RELEASE}" "${rocm_family}" \
    > "${ROCM_PATH}/.stamp"
fi

{
  printf 'export ROCM_RELEASE=%q\n' "${ROCM_RELEASE}"
  printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}"
} > "${env_out}"
echo "==> Wrote ${env_out}"
