# Sourced inside the container for the nccl.m2n (PyTorch) steps, instead of env.sh: rocm-env.sh puts
# _rocm_sdk_devel first in LD_LIBRARY_PATH, torch uses _rocm_sdk_core, and two HIP/LLVM copies abort.
S=/opt/venv/lib/python3.12/site-packages
export WORK=${WORK:-$HOME/m2n-ruby}
export SRC=${SRC:-$WORK/rocm-systems}
export NCCLX=${NCCLX:-$WORK/nccl-extensions}
export PATH=/opt/openmpi/bin:/opt/venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/lib:$S/_rocm_sdk_core/lib:$S/_rocm_sdk_core/lib/rocm_sysdeps/lib:$S/_rocm_sdk_core/lib/llvm/lib:/opt/hipdev
# Only for rocm/sitecustomize.py (not in the wheel): one librccl per process, the one libnccl_m2n links.
export PYTHONPATH=$NCCLX/python/rocm
export NCCL_M2N_LIBRARY=$WORK/mk/lib/libnccl_m2n.so NCCL_M2N_RCCL_LIBRARY=$WORK/rccl-install/lib/librccl.so.1
export NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_ENV_PLUGIN=none NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0} NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)} MASTER_PORT=${MASTER_PORT:-29533}
export SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_m2n} SSH_PORT=${SSH_PORT:-2222}
# MPI launcher for Python ranks; LD_PRELOAD goes to the ranks, not to mpirun. Prepend -np/--hostfile.
PYMPI_ARGS=(--map-by ppr:8:node --bind-to none --mca pml ob1 --mca btl ^vader,openib
  --mca oob_tcp_if_include $NCCL_SOCKET_IFNAME --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME
  --mca plm_rsh_agent "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_M2N_LIBRARY -x NCCL_M2N_RCCL_LIBRARY
  -x NCCL_GIN_ENABLE -x NCCL_GIN_TYPE -x NCCL_CUMEM_ENABLE -x NCCL_DMABUF_ENABLE -x HSA_NO_SCRATCH_RECLAIM
  -x NCCL_ENV_PLUGIN -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x MASTER_ADDR -x MASTER_PORT
  env LD_PRELOAD=$NCCL_M2N_RCCL_LIBRARY python)
