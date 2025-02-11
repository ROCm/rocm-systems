# -------------------------------------------------------------------------------------- #
#
# video decode tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME videodecode
    TARGET videodecode
    GPU ON
    ENVIRONMENT
        "${_base_environment};ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api"
    RUN_ARGS -i ${PROJECT_BINARY_DIR}/videos -t 1
    LABELS "videodecode")

rocprofiler_systems_add_validation_test(
    NAME videodecode-sampling
    PERFETTO_METRIC "rocm_rocdecode_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "videodecode"
    ARGS -l rocDecCreateVideoParser -c 2 -d 1 --counter-names "GPU VCN Activity")
