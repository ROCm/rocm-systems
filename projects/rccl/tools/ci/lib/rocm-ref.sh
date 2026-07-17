#!/usr/bin/env bash
# Reference-count the shared ROCm cache. References are acquired by fetch-rocm.sh
# (ROCM_REF_TOKEN) under a per-version lock and dropped here at end of run.
#
#   rocm-ref.sh release   Drop this run's reference for ROCM_RELEASE. The tree is
#                         kept warm; fetch-rocm.sh prunes unreferenced nightly/rc
#                         versions on the next version bump.
#
# Environment:
#   RCCL_DEVICE_API_CACHE   Cache root (default: /apps/rccl-ci)
#   ROCM_RELEASE            Concrete version (required; empty => no-op)
#   ROCM_REF_TOKEN          This run's reference token (e.g. <run_id>-<attempt>)

set -euo pipefail

op="${1:-}"
# "release-gc" is accepted as a legacy alias for "release".
if [[ "${op}" != "release" && "${op}" != "release-gc" ]]; then
  echo "usage: $0 release" >&2
  exit 2
fi

ref_token="${ROCM_REF_TOKEN:-}"
CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"

if [[ -z "${ROCM_RELEASE:-}" ]]; then
  echo "==> ROCM_RELEASE is empty; nothing to release"
  exit 0
fi

ROCM_PATH="${CACHE_DIR}/rocm/rocm-${ROCM_RELEASE}"
refs_dir="${ROCM_PATH}.refs"
lock_file="${CACHE_DIR}/downloads/.lock-${ROCM_RELEASE}"
mkdir -p "${CACHE_DIR}/downloads"

# Lock the version so we order against fetch-rocm.sh's acquire/prune.
exec {lock_fd}>"${lock_file}"
flock "${lock_fd}"

if [[ -n "${ref_token}" ]]; then
  rm -f "${refs_dir}/${ref_token}"
  echo "==> Released ROCm reference ${ref_token} (kept ${ROCM_PATH} warm)"
else
  echo "==> No ROCM_REF_TOKEN set; nothing to release"
fi

flock -u "${lock_fd}"
