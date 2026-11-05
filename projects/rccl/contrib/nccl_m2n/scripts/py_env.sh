# Sourced inside the container for the nccl.m2n (PyTorch) steps, instead of env.sh: rocm-env.sh puts
# _rocm_sdk_devel first in LD_LIBRARY_PATH, torch uses _rocm_sdk_core, and two HIP/LLVM copies abort.
S=/opt/venv/lib/python3.12/site-packages
export WORK=${WORK:-$HOME/m2n-ruby}
export SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
export M2N_PY=$SRC/projects/rccl/bindings/nccl_m2n
export PATH=/opt/openmpi/bin:/opt/venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/lib:$S/_rocm_sdk_core/lib:$S/_rocm_sdk_core/lib/rocm_sysdeps/lib:$S/_rocm_sdk_core/lib/llvm/lib:/opt/hipdev
# Workaround, see README "Why the extra environment": rocm/sitecustomize.py (test-only, not in the wheel)
# and LD_PRELOAD keep one librccl per process, the one libnccl_m2n (bundled in the wheel) links.
export PYTHONPATH=$M2N_PY/rocm
export NCCL_M2N_RCCL_LIBRARY=$WORK/rccl-install/lib/librccl.so.1
export NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_ENV_PLUGIN=none NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0} NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)} MASTER_PORT=${MASTER_PORT:-29533}
export SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_m2n} SSH_PORT=${SSH_PORT:-2222}
# MPI launcher for Python ranks; LD_PRELOAD goes to the ranks, not to mpirun. Prepend -np/--hostfile.
PYMPI_ARGS=(--map-by ppr:8:node --bind-to none --mca pml ob1 --mca btl ^vader,openib
  --mca oob_tcp_if_include $NCCL_SOCKET_IFNAME --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME
  --mca plm_rsh_agent "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_M2N_RCCL_LIBRARY
  -x NCCL_GIN_ENABLE -x NCCL_GIN_TYPE -x NCCL_CUMEM_ENABLE -x NCCL_DMABUF_ENABLE -x HSA_NO_SCRATCH_RECLAIM
  -x NCCL_ENV_PLUGIN -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x MASTER_ADDR -x MASTER_PORT
  env LD_PRELOAD=$NCCL_M2N_RCCL_LIBRARY python)
