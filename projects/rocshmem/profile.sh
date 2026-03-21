#!/bin/bash

rocprof-sys-instrument --mpi --output ./build/tests/functional_tests/rocshmem_functional_tests.inst \
  --log-file mylog.log --verbose --debug \
  --print-instrumented functions --error --min-instructions 1 \
  -- ./build/tests/functional_tests/rocshmem_functional_tests

mpirun -n 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=1 \
  -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=6442450944 \
  -x ROCSHMEM_TEST_UUID=1 --timeout 300 --map-by numa \
  /opt/rocm/bin/rocprof-sys-run \
  --rocm-hip-runtime-api-operations-annotate-backtrace hipLaunchKernel hipMemcpy hipMemcpyAsync \
  hipMalloc hipFree hipStreamSynchronize --config "./rocprof-sys.cfg" -- \
 ./build/tests/functional_tests/rocshmem_functional_tests.inst -a 26 -w 1 -z 64 \
 -s 4 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_4B.log
#  -s 1048576 2>&1 | tee ./tests-results/logs-heatmap-after/wgput_n2_w1_z64_1048576B.log
