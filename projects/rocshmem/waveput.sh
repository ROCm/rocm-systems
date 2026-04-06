#!/bin/bash

set -eux

export GPU_MAX_HW_QUEUES=32

LOG_DIR=./tests-results-waveput/logs-heatmap-loadstore-16-UNCACHED
mkdir -p $LOG_DIR

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 1 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w1_z64_1073741824B.log
  
mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=2 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 2 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w2_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=4 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 4 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w4_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=8 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 8 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w8_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=16 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 16 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w16_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=32 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 32 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w32_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=64 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 64 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w64_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=128 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 128 -z 64 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w128_z64_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 1 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w1_z1024_1073741824B.log
 
mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=2 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 2 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w2_z1024_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=4 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 4 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w4_z1024_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=8 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 8 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w8_z1024_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=16 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 16 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w16_z1024_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=32 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 32 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w32_z1024_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=64 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 64 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w64_z1024_1073741824B.log

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=128 \
 -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
 -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
 ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 128 -z 1024 \
 -v 1073741824 2>&1 | tee $LOG_DIR/waveput_n2_w128_z1024_1073741824B.log

# mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=16 \
#  -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
#  -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
#  ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 16 -z 1024 \
#  -s 1073741824 2>&1 | tee ./tests-results/logs-heatmap-loadstore-fix/waveput_n2_w16_z1024_1073741824B.log

# mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
#  -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=68719476736 \
#  -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
#  ./build/tests/functional_tests/rocshmem_functional_tests -a 31 -w 1 -z 64 \
#  -s 1024 -n 1 -nskip 0 2>&1 | tee ./tests-results/logs-heatmap-after/waveput_n2_w1_z64_1024B.log

#  -s 8 2>&1 | tee ./tests-results/logs-heatmap-after/waveput_n2_w16_z64_8B.log
#  -s 1048576 2>&1 | tee ./tests-results/logs-heatmap-after/waveput_n2_w2_z64_1048576B.log
#  -s 1024 2>&1 | tee ./tests-results/logs-heatmap-after/waveput_n2_w1_z64_64B.log
#  -s 1024 2>&1 | tee ./tests-results/logs-heatmap-after/waveput_n2_w1_z64_1024.log
