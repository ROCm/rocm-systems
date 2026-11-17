# Native (no container) env: CI ROCm 10.1.0 (TheRock 5e21fb213299) + CI OpenMPI 5.0.9.
# Slurm propagates the login node soft nofile (1024); 2-node+ GIN/DevComm setup needs far more.
ulimit -n $(ulimit -Hn) 2>/dev/null
export ROCM_PATH=/apps/rccl-ci/nightly/ruby64-rccl-perf-nightly/build/rocm-10.1-ga/10.1/5e21fb213299/rocm
export MPI_HOME=${MPI_HOME:-/apps/rccl-ci/ompi/install/5.0.9}
export WORK=${WORK:-$HOME/m2n-ruby/ci101}
export SRC=${SRC:-$HOME/m2n-ruby/rocm-systems}
export RCCL_INSTALL=$WORK/rccl-install
export BIN=$WORK/mk/bin
export PATH=$MPI_HOME/bin:$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$RCCL_INSTALL/lib:$WORK/mk/lib:$MPI_HOME/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}
export NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=${NCCL_GIN_TYPE:-2} NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1
export HSA_NO_SCRATCH_RECLAIM=1 NCCL_ENV_PLUGIN=none NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_rsa}   # native: default cluster key; id_m2n is only authorized inside the container
export SSH_PORT=${SSH_PORT:-22} CMAKE_GENERATOR=${CMAKE_GENERATOR:-Unix Makefiles}
mkdir -p $WORK/logs
