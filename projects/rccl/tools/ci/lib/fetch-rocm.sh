#!/usr/bin/env bash
# Fetch + extract a ROCm tree for the device-API CI job, using TheRock's own
# installer (build_tools/install_rocm_from_artifacts.py) rather than a
# hand-rolled wget/tar.
#
# The installer (and its transitive imports) live in TheRock, so we clone
# TheRock at a pinned ref and build a venv from its requirements.txt once,
# cached on the shared NFS path. The flattened nightly tarball
# (therock-dist-<os>-<family>-<version>.tar.gz) is downloaded + extracted into
# a version-keyed cache dir.
#
# Result: writes $WORKDIR/.ci-out/rocm.env exporting the concrete ROCM_PATH and
# ROCM_RELEASE so the build-ompi / RCCL / run steps all use the same tree.
#
# Environment:
#   RCCL_DEVICE_API_CACHE  Persistent cache root (default: /apps/rccl-ci)
#   ROCM_RELEASE           ROCm nightly/dev version (e.g. 7.13.0a20260515).
#                          Empty => resolve the latest nightly for the family.
#   ROCM_AMDGPU_FAMILY     TheRock artifact group (default: gfx950-dcgpu)
#   ROCM_PATH_OVERRIDE     Use this existing ROCm tree directly; skips fetching.
#                          An ambient ROCM_PATH is intentionally ignored, so a
#                          stray `module load rocm` on the node cannot silently
#                          replace the pinned nightly with /opt/rocm.
#   THEROCK_REPO           TheRock git URL (default: ROCm/TheRock)
#   THEROCK_REF            TheRock pinned ref for the installer tool
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)

set -euxo pipefail

rocm_family="${ROCM_AMDGPU_FAMILY:-gfx950-dcgpu}"
THEROCK_REPO="${THEROCK_REPO:-https://github.com/ROCm/TheRock.git}"
# Pinned to the same TheRock commit the in-repo RCCL CI workflows check out.
THEROCK_REF="${THEROCK_REF:-910bc0bc9e25ac533f100b343ac03b30562a899a}"

WORKDIR="${RCCL_DEVICE_API_WORKDIR:-${WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
fi

CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"
if ! mkdir -p "${CACHE_DIR}/rocm" "${CACHE_DIR}/downloads" "${CACHE_DIR}/therock"; then
  echo "ERROR: cannot create cache dirs under ${CACHE_DIR} (is /apps writable?)" >&2
  exit 1
fi

env_out="${WORKDIR}/.ci-out/rocm.env"
mkdir -p "${WORKDIR}/.ci-out"

# Explicit override only: a dedicated var, NOT an ambient ROCM_PATH (which the
# node may export via `module load rocm`). This keeps us from silently building
# against /opt/rocm instead of the pinned nightly.
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

# Ambient ROCM_PATH (e.g. from the node environment) is deliberately discarded;
# the fetched/cached nightly path is computed below.
unset ROCM_PATH

# ---------------------------------------------------------------------------
# TheRock installer (clone + venv), cached and keyed on the pinned ref.
# ---------------------------------------------------------------------------
THEROCK_DIR="${CACHE_DIR}/therock/${THEROCK_REF}"
venv="${THEROCK_DIR}/.venv"
THEROCK_PY="${venv}/bin/python"
deps_stamp="${venv}/.deps-stamp"
want_stamp="ref=${THEROCK_REF}"

if [[ -x "${THEROCK_PY}" ]] \
   && [[ "$(cat "${deps_stamp}" 2>/dev/null || true)" == "${want_stamp}" ]]; then
  echo "==> Reusing TheRock installer venv at ${venv}"
else
  echo "==> Provisioning TheRock (${THEROCK_REF}) installer into ${THEROCK_DIR}"
  if [[ ! -d "${THEROCK_DIR}/.git" ]]; then
    rm -rf "${THEROCK_DIR}"
    git clone --filter=blob:none "${THEROCK_REPO}" "${THEROCK_DIR}"
  fi
  if ! git -C "${THEROCK_DIR}" checkout -q "${THEROCK_REF}" 2>/dev/null; then
    git -C "${THEROCK_DIR}" fetch --filter=blob:none origin "${THEROCK_REF}"
    git -C "${THEROCK_DIR}" checkout -q "${THEROCK_REF}"
  fi
  rm -rf "${venv}"
  python3 -m venv "${venv}"
  "${THEROCK_PY}" -m pip install --quiet --upgrade pip
  "${THEROCK_PY}" -m pip install --quiet -r "${THEROCK_DIR}/requirements.txt"
  printf '%s\n' "${want_stamp}" > "${deps_stamp}"
fi

# ---------------------------------------------------------------------------
# Resolve a concrete version (so build + run agree and the cache key is stable).
# ---------------------------------------------------------------------------
if [[ -z "${ROCM_RELEASE:-}" ]]; then
  echo "==> ROCM_RELEASE unset; resolving latest nightly for ${rocm_family}"
  dry="$(cd "${THEROCK_DIR}/build_tools" \
    && "${THEROCK_PY}" install_rocm_from_artifacts.py \
        --latest-release --amdgpu-family "${rocm_family}" --dry-run 2>&1)" || {
    echo "ERROR: failed to resolve latest nightly for ${rocm_family}:" >&2
    printf '%s\n' "${dry}" >&2
    exit 1
  }
  ROCM_RELEASE="$(sed -n 's/.*Found latest release: *\([^ ]*\).*/\1/p' <<<"${dry}" | head -1)"
  [[ -n "${ROCM_RELEASE}" ]] \
    || ROCM_RELEASE="$(sed -n 's/.*(version \([^)]*\)).*/\1/p' <<<"${dry}" | head -1)"
  if [[ -z "${ROCM_RELEASE}" ]]; then
    echo "ERROR: could not parse a version from --dry-run output:" >&2
    printf '%s\n' "${dry}" >&2
    exit 1
  fi
  echo "==> Latest nightly: ${ROCM_RELEASE}"
fi

ROCM_PATH="${CACHE_DIR}/rocm/rocm-${ROCM_RELEASE}"

# ---------------------------------------------------------------------------
# Fetch on cache miss only (the installer wipes + recreates its output dir).
# ---------------------------------------------------------------------------
if [[ -x "${ROCM_PATH}/bin/hipcc" ]] \
   && grep -qx "family=${rocm_family}" "${ROCM_PATH}/.stamp" 2>/dev/null; then
  echo "==> Reusing cached ROCm ${ROCM_RELEASE} at ${ROCM_PATH}"
else
  echo "==> Fetching ROCm ${ROCM_RELEASE} (${rocm_family}) -> ${ROCM_PATH}"
  ( cd "${THEROCK_DIR}/build_tools" \
    && "${THEROCK_PY}" install_rocm_from_artifacts.py \
        --release "${ROCM_RELEASE}" \
        --amdgpu-family "${rocm_family}" \
        --output-dir "${ROCM_PATH}" )
  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: fetched ROCm at ${ROCM_PATH} has no bin/hipcc" >&2
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
