#!/bin/bash
CURRENT_DIR="$( dirname -- "$0"; )";

echo -e "Running Samples"

export ROCPROFILER_METRICS_PATH=${CURRENT_DIR}/../counters/derived_counters.xml

echo -e "\tProfiler Samples:"

# echo -e "\t\tApplication Replay Sample:"
# eval ${CURRENT_DIR}/profiler_application_replay

echo -e "\t\tKernel Replay Sample:"
eval ${CURRENT_DIR}/profiler_kernel_replay

# echo -e "\t\tUser Replay Sample:"
# eval ${CURRENT_DIR}/profiler_user_replay

echo -e "\t\tDevice Profiling Sample:"
eval ${CURRENT_DIR}/profiler_device_profiling



# echo -e "\tTracer Samples:"


# echo -e "\t\tHIP/HSA Trace Sample:"
# eval ${CURRENT_DIR}/tracer_hip_hsa
