#!/bin/bash

set -eux

# export GPU_DEVICE_ORDINAL="0"

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 26 -w 1 -z 1024 \
 -s 1073741824 2>&1 | tee ./tests-results/logs-heatmap-loadstore/wgput_n2_w1_z1024_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 26 -w 1 -z 64 \
 -s 1073741824 2>&1 | tee ./tests-results/logs-heatmap-loadstore/wgput_n2_w1_z64_1073741824B.log
  
mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=2 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 26 -w 2 -z 64 \
 -s 1073741824 2>&1 | tee ./tests-results/logs-heatmap-loadstore/wgput_n2_w2_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=16 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 26 -w 16 -z 1024 \
 -s 1073741824 2>&1 | tee ./tests-results/logs-heatmap-loadstore/wgput_n2_w16_z1024_1073741824B.log


# mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
#  -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
#  -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
#  ./build/tests/functional_tests/rocshmem_functional_tests -a 26 -w 1 -z 64 \
#  -s 1024 -n 1 -nskip 0 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_1024B.log

#  -s 8 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w16_z64_8B.log
#  -s 1048576 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w2_z64_1048576B.log
#  -s 1024 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_64B.log
#  -s 1024 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_1024.log
