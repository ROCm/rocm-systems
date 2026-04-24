#!/bin/bash

set -eux

# export GPU_MAX_HW_QUEUES=32
export HIP_VISIBLE_DEVICES=1,2,5,6,7

# Test cases
WGGET_TEST="24" 
WGGET_NBI_TEST="25" 

WGPUT_TEST="26" 
WGPUT_NBI_TEST="27"

WAVEPUT_TEST="30"
WAVEPUT_NBI_TEST="31"

TEST_CASE=$WGPUT_TEST
TEST_NAME="wgput"

LOG_DIR=./tests-results-skip0/logs-heatmap-${TEST_NAME}-narrow-fence
mkdir -p $LOG_DIR

NP=2
MPI_FLAGS="-np ${NP} --mca pml ucx --mca osc ucx --timeout 300 --map-by numa"
ROCSHMEM_FLAGS="-x ROCSHMEM_HEAP_SIZE=68719476736 -x ROCSHMEM_TEST_UUID=1"
UCX_FLAGS="-x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384"

threads_list=(1 4 16 64 256 1024)
workgroups_list=(1 2 4 8 16 32 64 128)

###############################################################################
# Small tests
###############################################################################
workgroups=1
threads_list=(1 4 16 64 256 1024)
max_size=1048576

for threads in ${threads_list[@]}; do
  mpirun $MPI_FLAGS $UCX_FLAGS $ROCSHMEM_FLAGS -x ROCSHMEM_MAX_NUM_CONTEXTS=$workgroups\
    ./build/tests/functional_tests/rocshmem_functional_tests -a $TEST_CASE -w $workgroups -z $threads \
    -s $max_size -n 50 -nskip 0 2>&1 | tee $LOG_DIR/${TEST_NAME}_n${NP}_w${workgroups}_z${threads}_${max_size}B.log
done

###############################################################################
# Large tests
###############################################################################
threads_list=(64 256 1024)
workgroups_list=(1 2 4 8 16 32 64 128)
max_size=1073741824
for workgroups in ${workgroups_list[@]}; do
  for threads in ${threads_list[@]}; do
    mpirun $MPI_FLAGS $UCX_FLAGS $ROCSHMEM_FLAGS -x ROCSHMEM_MAX_NUM_CONTEXTS=$workgroups\
      ./build/tests/functional_tests/rocshmem_functional_tests -a $TEST_CASE -w $workgroups -z $threads \
      -v $max_size -n 50 -nskip 0 2>&1 | tee $LOG_DIR/${TEST_NAME}_n${NP}_w${workgroups}_z${threads}_${max_size}B.log
  done
done

###############################################################################
# Special tests
###############################################################################
workgroups=72
threads=512
max_size=1073741824
TEST_CASE=$WAVEPUT_NBI_TEST

mpirun $MPI_FLAGS $UCX_FLAGS $ROCSHMEM_FLAGS -x ROCSHMEM_MAX_NUM_CONTEXTS=$workgroups\
      ./build/tests/functional_tests/rocshmem_functional_tests -a $TEST_CASE -w $workgroups -z $threads \
      -v $max_size -n 50 -nskip 0 2>&1 | tee $LOG_DIR/${TEST_NAME}_n${NP}_w${workgroups}_z${threads}_${max_size}B.log

# workgroups=32
# threads=64
# max_size=$((8*1024*1024))
# TEST_CASE=$WAVEPUT_NBI_TEST

# mpirun $MPI_FLAGS $UCX_FLAGS -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_MAX_NUM_CONTEXTS=$workgroups\
#       ./build/tests/functional_tests/rocshmem_functional_tests -a $TEST_CASE -w $workgroups -z $threads \
#       -v $max_size -n 5000 -nskip 10 2>&1 | tee $LOG_DIR/${TEST_NAME}_n${NP}_w${workgroups}_z${threads}_${max_size}B.log

# workgroups=1
# threads=1
# max_size=$((8*1024*1024))
# TEST_CASE=$WAVEPUT_NBI_TEST

# mpirun $MPI_FLAGS $UCX_FLAGS -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_MAX_NUM_CONTEXTS=$workgroups\
#       ./build/tests/functional_tests/rocshmem_functional_tests -a $TEST_CASE -w $workgroups -z $threads \
#       -v $max_size -n 5000 -nskip 10 2>&1 | tee $LOG_DIR/${TEST_NAME}_n${NP}_w${workgroups}_z${threads}_${max_size}B.log
