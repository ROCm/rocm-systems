#!/bin/bash

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=6442450944 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 26 -w 1 -z 64 \
 -s 1048576 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_1048576B.log
#  -s 1024 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_64B.log
#  -s 1024 -n 1 -nskip 0 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_64B.log
#  -s 4 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_4B.log
#  -s 1024 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_1024.log
