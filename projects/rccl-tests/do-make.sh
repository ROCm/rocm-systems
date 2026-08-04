#!/bin/bash
B=/work1/lmeadows/rccl-work/rccl/projects/rccl/build/release
# Change ARCH as needed
ARCH=gfx950
make MPI=1 MPI_HOME=/work1/lmeadows/openmpi \
  ROCM_PATH=/work/lmeadows/rocm/srock \
  NCCL_HOME=$B \
  CUSTOM_NCCL_LIB=$B/librccl.so \
  GPU_TARGETS=$ARCH
