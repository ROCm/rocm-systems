#!/bin/bash
CURRENT_DIR=$(dirname -- $(realpath ${BASH_SOURCE[0]}));
ROCM_PATH="${ROCM_PATH:=/opt/rocm}"


echo -e "Running Samples"

export ROCPROFILER_METRICS_PATH=${ROCM_PATH}/libexec/rocprofiler/counters/derived_counters.xml

echo -e "\tProfiler Samples:"

echo -e "\t\tKernel Replay Sample:"
eval ${CURRENT_DIR}/profiler_kernel_replay


echo -e "\t\tDevice Profiling Sample:"
eval ${CURRENT_DIR}/profiler_device_profiling


echo -e "\tTracer Samples:"

echo -e "\t\tHIP/HSA Trace Synchronous Sample:"
eval ${CURRENT_DIR}/tracer_hip_hsa

echo -e "\t\tHIP/HSA Trace ASynchronous Sample:"
eval ${CURRENT_DIR}/tracer_hip_hsa_async