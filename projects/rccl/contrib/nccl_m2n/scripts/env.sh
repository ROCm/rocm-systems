# Sourced inside the container by the build/run scripts (the Python checks use py_env.sh instead).
# fenic0, port 2222 and ~/m2n-ruby are example defaults for Ruby; override NCCL_SOCKET_IFNAME, SSH_PORT, WORK.
# M2N_ENV=<file> replaces the container defaults with another stack (e.g. runbook/native/env-ci-rocm-10.1.sh).
if [[ -n ${M2N_ENV:-} ]]; then
  . "$M2N_ENV"
else
  . /etc/profile.d/rocm-env.sh
  export WORK=${WORK:-$HOME/m2n-ruby}
  export SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
  export RCCL_INSTALL=$WORK/rccl-install
  export BIN=$WORK/mk/bin
  export LD_LIBRARY_PATH=$RCCL_INSTALL/lib:$WORK/mk/lib:$LD_LIBRARY_PATH
  export NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1
  export HSA_NO_SCRATCH_RECLAIM=1 NCCL_ENV_PLUGIN=none NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0}
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_m2n} SSH_PORT=${SSH_PORT:-2222}   # sshd of the container
  export MPI_HOME=${MPI_HOME:-/opt/openmpi} CMAKE_GENERATOR=${CMAKE_GENERATOR:-Ninja}
  mkdir -p $WORK/logs
fi
# ssh into the other nodes' sshd: mpirun's launch agent and remote cleanup (pkill).
M2N_SSH="ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
# mpirun with the flags every launcher shares: "${M2N_MPIRUN[@]}" -np <N> [--hostfile <f>] <binary> [args].
M2N_MPIRUN=(mpirun --map-by ppr:8:node --bind-to none --mca pml ob1 --mca btl ^vader,openib
  --mca oob_tcp_if_include $NCCL_SOCKET_IFNAME --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME
  --mca plm_rsh_agent "$M2N_SSH"
  -x PATH -x LD_LIBRARY_PATH -x NCCL_GIN_ENABLE -x NCCL_GIN_TYPE -x NCCL_CUMEM_ENABLE -x NCCL_DMABUF_ENABLE
  -x HSA_NO_SCRATCH_RECLAIM -x NCCL_ENV_PLUGIN -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG
  -x NCCL_RESHARD_ALGORITHM -x NCCL_RESHARD_LOG_LEVEL -x NCCL_M2N_ENABLE_EXPERIMENTAL)
