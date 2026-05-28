#!/bin/bash

SEQUENCE_LENGTHS=(50 128 256 512 1024 2048 4096)
ALL_REDUCE_ALGOS=(1 2 3 4)

HIP_DEV_FORCE_KERNARG=1

for SEQ_LEN in "${SEQUENCE_LENGTHS[@]}"; do
	for ALGO in "${ALL_REDUCE_ALGOS[@]}"; do
		echo "Running sequence length $SEQ_LEN with intra-node all_reduce $ALGO"
		ENABLE_INTRA_NODE_COMM=1 rocprofv3 --hip-trace --kernel-trace --stats --output-format PFTRACE -d rocprof_trace/intranode_input"$SEQ_LEN"_allreduce"$ALGO"/ -- torchrun --nproc_per_node=8 all_reduce.py --sequence_lengths $SEQ_LEN --all_reduce $ALGO --tracing
	done
done
