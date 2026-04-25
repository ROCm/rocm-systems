#!/bin/bash
# Mellanox ConnectX environment variables
# Sourced in every shell inside the container.
#
# Host RDMA libraries (libibverbs, libmlx5) are automatically bind-mounted
# by mnctl when InfiniBand devices are detected. No driver installation is
# typically needed.
#
# Uncomment and adjust variables below for your cluster:

# export NCCL_IB_HCA=mlx5
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7
