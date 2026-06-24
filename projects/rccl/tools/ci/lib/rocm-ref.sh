#!/usr/bin/env bash
# Reference-count the shared nightly ROCm cache so the last run out GCs it.
# References are acquired by fetch-rocm.sh (ROCM_REF_TOKEN) under the same lock.
#
#   rocm-ref.sh release-gc   Drop this run's reference; delete the tree if none
#                            remain and the channel is nightly.
#
# Environment:
#   RCCL_DEVICE_API_CACHE   Cache root (default: /apps/rccl-ci)
#   ROCM_RELEASE_CHANNEL    nightly|dev|prerelease|release (default: nightly)
#   ROCM_RELEASE            Concrete version (required; empty => no-op)
#   ROCM_AMDGPU_FAMILY      Artifact family (default: gfx950-dcgpu)
#   ROCM_REF_TOKEN          This run's reference token (e.g. <run_id>-<attempt>)

set -euo pipefail

op="${1:-}"
if [[ "${op}" != "release-gc" ]]; then
  echo "usage: $0 release-gc" >&2
  exit 2
fi

rocm_family="${ROCM_AMDGPU_FAMILY:-gfx950-dcgpu}"
rocm_channel="${ROCM_RELEASE_CHANNEL:-nightly}"
ref_token="${ROCM_REF_TOKEN:-}"
CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"

# Guard against a destructive rm on an empty version (e.g. provision failed).
if [[ -z "${ROCM_RELEASE:-}" ]]; then
  echo "==> ROCM_RELEASE is empty; nothing to release"
  exit 0
fi

ROCM_PATH="${CACHE_DIR}/rocm/rocm-${ROCM_RELEASE}"
refs_dir="${ROCM_PATH}.refs"
lock_file="${CACHE_DIR}/downloads/.lock-${rocm_family}-${rocm_channel}-${ROCM_RELEASE}"
mkdir -p "${CACHE_DIR}/downloads"

exec {lock_fd}>"${lock_file}"
flock "${lock_fd}"

# Drop our own reference (idempotent).
if [[ -n "${ref_token}" ]]; then
  rm -f "${refs_dir}/${ref_token}"
  echo "==> Released ROCm reference ${ref_token}"
fi

# Only nightly builds are garbage-collected; GA/release/etc. stay cached.
if [[ "${rocm_channel}" != "nightly" ]]; then
  echo "==> Channel '${rocm_channel}' is durable; keeping ${ROCM_PATH}"
  flock -u "${lock_fd}"
  exit 0
fi

# Any remaining reference files mean another run still needs this build.
remaining=0
if [[ -d "${refs_dir}" ]]; then
  remaining="$(find "${refs_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d ' ')"
fi

if [[ "${remaining}" -gt 0 ]]; then
  echo "==> ${remaining} reference(s) still using ${ROCM_PATH}; keeping it"
else
  echo "==> No references remain; removing nightly ROCm ${ROCM_PATH}"
  rm -rf "${ROCM_PATH}" "${refs_dir}"
fi

flock -u "${lock_fd}"
