#!/bin/bash
# 2-node OpenMPI bootstrap smoke for the RCCL AINIC Crusoe probe.
#
# Submitted by scripts/crusoe_probe_lib.sh (mpi_smoke_pair) as a short sbatch job
# pinned to a candidate symmetric pair, BEFORE that pair is handed to build+test.
# It reproduces the exact launch path the real suites use -- OpenMPI `--mca plm
# slurm` driven through the scripts/spur_srun.sh shim, with OOB and TCP BTL
# pinned to the management interface (PROBE_OOB_IF, default ens3) -- and simply
# runs `hostname` on one rank per node. If mpirun cannot bring its daemons up
# across the two nodes (e.g. "an ORTE daemon has unexpectedly failed ... lack of
# common network interfaces and/or no route found between them"), the pair is
# rejected and re-probed instead of failing every test at run time.
#
# Prints "MPI_SMOKE_OK <hosts>" on success, "MPI_SMOKE_FAIL ..." otherwise.
#
# Inputs (forwarded via sbatch --export=ALL): GITHUB_WORKSPACE, TEST_RUNNER_DIR,
# RUNNER_TEMP, PROBE_OOB_IF.
set -o pipefail
export PATH="/opt/openmpi/bin:/opt/rocm/bin:/usr/sbin:/sbin:/usr/bin:/bin:${PATH:-}"
export LD_LIBRARY_PATH="/opt/ucx/lib:/opt/openmpi/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
oob_if="${PROBE_OOB_IF:-ens3}"

cd "${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}" 2>/dev/null || true

# Hosts in this allocation (short names), same helper the run batch uses.
HOSTS=$(python3 -c 'import os
from lib.test_executor import expand_slurm_hostlist
nl = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST") or ""
print(",".join(expand_slurm_hostlist(nl)))' 2>/dev/null)
first="${HOSTS%%,*}"

# SPUR runs this batch script on every allocated node. Elect the first host as
# the mpirun launcher; the others park on an NFS sentinel so the job does not
# leave COMPLETING before the launcher is done (mirrors crusoe_run_batch.sh).
_done="${RUNNER_TEMP}/mpi_smoke_done_${SLURM_JOB_ID:-$$}"
if [ -n "${first}" ] && [ "$(hostname -s)" != "${first}" ]; then
  end=$(( $(date +%s) + 180 ))
  while [ ! -f "${_done}" ] && [ "$(date +%s)" -lt "${end}" ]; do sleep 3; done
  exit 0
fi
trap 'rm -f "${PARK_SENTINEL:-}"; touch "${_done}"' EXIT

# Local session dir on tmpfs, same as crusoe_run_batch.sh: $HOME is NFS and
# OpenMPI's session/tmp there can make orted fail at launch, which would look
# exactly like a network failure -- keep the smoke faithful to the real run.
export TMPDIR="/dev/shm/rccl-smoke-${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
# 1 task/node allocation; OpenMPI plm slurm reads this. The shim maps ranks.
if [ -n "${SLURM_NNODES:-}" ]; then
  export SLURM_TASKS_PER_NODE="1(x${SLURM_NNODES})"
fi
# SPUR srun != Slurm srun: point OpenMPI plm slurm at the in-tree shim.
SHIMDIR="/dev/shm/rccl-srun-smoke-${SLURM_JOB_ID:-$$}"
mkdir -p "${SHIMDIR}"
if ! cp "${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}/scripts/spur_srun.sh" "${SHIMDIR}/srun" 2>/dev/null; then
  echo "MPI_SMOKE_FAIL could not stage srun shim"
  exit 3
fi
chmod +x "${SHIMDIR}/srun"
export PATH="${SHIMDIR}:${PATH}"
export OMPI=/opt/openmpi
export PARK_SENTINEL="/dev/shm/crusoe-mpirun-smoke.active"
touch "${PARK_SENTINEL}"

nnodes=$(printf '%s\n' "${HOSTS}" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "${nnodes}" -lt 1 ]; then
  echo "MPI_SMOKE_FAIL empty host list"
  exit 2
fi
hostspec=$(printf '%s' "${HOSTS}" | awk -F, '{for(i=1;i<=NF;i++){printf "%s%s:1",(i>1?",":""),$i}}')

echo "MPI smoke: mpirun -np ${nnodes} --host ${hostspec} (oob=${oob_if}) on ${HOSTS}"
out=$(timeout -k 5 90 mpirun -np "${nnodes}" --host "${hostspec}" --map-by ppr:1:node \
  --bind-to none --oversubscribe \
  --prefix /opt/openmpi \
  --mca plm slurm --mca orte_tmpdir_base /dev/shm \
  --mca pml ob1 --mca osc ^ucx \
  --mca btl tcp,self \
  --mca btl_tcp_if_include "${oob_if}" \
  --mca oob_tcp_if_include "${oob_if}" \
  hostname 2>&1)
rc=$?
printf '%s\n' "${out}"

pat=$(printf '%s' "${HOSTS}" | sed 's/,/|/g')
matched=$(printf '%s\n' "${out}" | grep -aE "^(${pat})$" | sort -u | wc -l)
if [ "${rc}" -eq 0 ] && [ "${matched}" -ge "${nnodes}" ]; then
  echo "MPI_SMOKE_OK ${HOSTS}"
  exit 0
fi
echo "MPI_SMOKE_FAIL rc=${rc} matched=${matched}/${nnodes} oob=${oob_if}"
exit 1
