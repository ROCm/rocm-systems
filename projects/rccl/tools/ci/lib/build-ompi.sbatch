#!/usr/bin/env bash
#SBATCH --job-name=rccl-ompi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=interactive
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Build GPU-aware OpenMPI once, as a prereq for the gin / device-api legs, so the
# two 8-GPU legs share one warm cache instead of each rebuilding it (or racing on
# the shared install path). Invoked via sbatch --wait from rccl-ci.yml after ROCm
# is provisioned. No GPU needed: this is a CPU compile against the ROCm headers.

set -euo pipefail

echo "=== sbatch build-ompi on $(hostname) ==="
date -Is

workdir="${RCCL_DEVICE_API_WORKDIR:?RCCL_DEVICE_API_WORKDIR must be exported from the workflow}"
cd "${workdir}"
ci_dir="${workdir}/projects/rccl/tools/ci"

# ROCm is provisioned on the head node; consume its shared tree via rocm.env.
rocm_env="${workdir}/.ci-out/rocm.env"
if [[ ! -f "${rocm_env}" ]]; then
  echo "ERROR: ${rocm_env} not found; provision ROCm before build-ompi.sbatch" >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null
source "${rocm_env}"
set +a

if [[ -z "${RCCL_BUILD_JOBS:-}" ]]; then
  nproc_count="$(nproc)"
  RCCL_BUILD_JOBS=$(( nproc_count < 32 ? nproc_count : 32 ))
fi
export RCCL_BUILD_JOBS

"${ci_dir}/lib/build-ompi.sh"
