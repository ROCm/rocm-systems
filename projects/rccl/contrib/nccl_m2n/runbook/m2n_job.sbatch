#!/bin/bash
#SBATCH -p meta64
#SBATCH --exclusive
#SBATCH -t 16:00:00
#SBATCH -o %x-%j.out
#SBATCH --signal=B:USR1@300
# Host-side wrapper for the bench phases: image per node (built if missing), container $NAME on every
# node, run $PHASES_SCRIPT (phases.sh or model_phases.sh) inside the container on the first node,
# then remove the containers. Pass PHASES / MODEL_RUNS through the environment, NOT through
# --export=VAR=... (sbatch splits --export values at commas): PHASES="..." sbatch --export=ALL ...
set -u
RB=${RB:-$SLURM_SUBMIT_DIR}   # submit from the runbook directory
W=${WORK:-$HOME/m2n-ruby}
NAME=${NAME:-m2n-multi}
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
N1=${NODES[0]}
mkdir -p $W/logs/bench
HF=$W/logs/bench/hostfile-$SLURM_JOB_ID
printf '%s.rck.dcgpu slots=8\n' "${NODES[@]}" > $HF
echo "job $SLURM_JOB_ID nodes: ${NODES[*]} script: ${PHASES_SCRIPT:-phases.sh} phases: ${PHASES:-} ${MODEL_RUNS:-} $(date -u)"

# Containers keep the GPUs busy after the job ends unless removed: clean up on exit, scancel (TERM) and
# 300 s before the time limit (USR1, --signal above).
RC=0
cleanup() {
  trap - EXIT TERM INT USR1
  for n in "${NODES[@]}"; do ssh $n "sudo docker rm -f $NAME >/dev/null 2>&1; echo \$(hostname -s) container removed"; done
  echo "JOB_DONE rc=$RC $(date -u)"
  exit $RC
}
trap cleanup EXIT
trap 'RC=143; cleanup' TERM INT
trap 'RC=124; echo "time limit close, stopping"; cleanup' USR1

for n in "${NODES[@]}"; do
  ssh -o StrictHostKeyChecking=no $n "systemctl is-active -q docker || sudo systemctl start docker
    sudo docker image inspect m2n-ruby:primus-rdma61 >/dev/null 2>&1 || { cd $RB && sudo docker build --network host \
      -t m2n-ruby:primus-rdma61 . > $W/logs/bench/docker_build_\$(hostname -s).log 2>&1; }
    df -h / | tail -1; NAME=$NAME bash $RB/start-m2n-container.sh 2>&1 | tail -1" &
done
wait
for n in "${NODES[@]}"; do ssh $n "sudo docker ps --format '{{.Names}}' | grep -qx $NAME" || { echo "NO CONTAINER on $n"; RC=1; }; done
if (( RC == 0 )); then
  # Background + wait so the traps fire while the phases run.
  ssh $N1 "sudo docker exec -u $(id -u) -e HOME=$HOME -e WORK=$W -e PHASES='${PHASES:-}' -e MODEL_RUNS='${MODEL_RUNS:-}' -e HF=$HF \
    -e SIZES='${SIZES:-}' -e ROUNDS='${ROUNDS:-}' -e BATCH_RUNS='${BATCH_RUNS:-}' -e NUM_TENSORS='${NUM_TENSORS:-}' \
    -e COMMS='${COMMS:-}' -e COPIES='${COPIES:-}' $NAME bash -lc 'bash $RB/${PHASES_SCRIPT:-phases.sh}'" &
  wait $! || RC=$?
fi
