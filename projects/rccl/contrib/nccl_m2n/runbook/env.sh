# Sourced inside the m2n-multi container by every build/run script.
. /etc/profile.d/rocm-env.sh
export WORK=${WORK:-$HOME/m2n-ruby}
export SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
export RCCL_INSTALL=$WORK/rccl-install
export BIN=$WORK/mk/bin
export LD_LIBRARY_PATH=$RCCL_INSTALL/lib:$WORK/mk/lib:$LD_LIBRARY_PATH
export NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1
export HSA_NO_SCRATCH_RECLAIM=1 NCCL_ENV_PLUGIN=none NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_m2n}
mkdir -p $WORK/logs
