# Sourced inside the container by the build/run scripts (the Python checks use py_env.sh instead).
# fenic0, port 2222 and ~/m2n-ruby are example defaults for Ruby; override NCCL_SOCKET_IFNAME, SSH_PORT, WORK.
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
mkdir -p $WORK/logs
