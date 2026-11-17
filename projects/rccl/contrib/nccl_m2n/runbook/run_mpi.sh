#!/bin/bash
# MPI run, 8 ranks per node. 1node runs on the current node; 2node uses $WORK/hostfile.
# Run inside the container on the first node.
# Usage: run_mpi.sh <label> <np> <1node|2node> [test args...]
. "$(dirname "$0")/env.sh"
L=$1; NP=$2; LAY=$3; shift 3
LOG=$WORK/logs/$L.log
PLACE=()
[[ $LAY == 2node ]] && PLACE=(--hostfile ${HF:-$WORK/hostfile})
{ echo "=== $L np=$NP $LAY host=$(hostname -s) $(date -u) args: $*"
  timeout ${TMO:-3600} mpirun -np $NP "${PLACE[@]}" --map-by ppr:8:node --bind-to none \
    --mca pml ob1 --mca btl ^vader,openib \
    --mca oob_tcp_if_include $NCCL_SOCKET_IFNAME --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME \
    --mca plm_rsh_agent "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
    -x PATH -x LD_LIBRARY_PATH -x NCCL_GIN_ENABLE -x NCCL_GIN_TYPE -x NCCL_CUMEM_ENABLE -x NCCL_DMABUF_ENABLE \
    -x HSA_NO_SCRATCH_RECLAIM -x NCCL_ENV_PLUGIN -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG \
    $BIN/basic_api_test_mpi "$@" --gtest_color=no
  echo "=== EXIT $? $(date -u)"; } > $LOG 2>&1
grep -E "^\[  PASSED|^\[  FAILED  \] [0-9]|^\[  SKIPPED \] [0-9]|^=== EXIT" $LOG | sort | uniq -c
exit $(grep '^=== EXIT' $LOG | cut -d' ' -f3)
