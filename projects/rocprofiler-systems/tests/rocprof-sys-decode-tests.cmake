# -------------------------------------------------------------------------------------- #
#
# video decode tests
#
# -------------------------------------------------------------------------------------- #

set(_decode_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api"
    "ROCPROFSYS_ROCM_SMI_METRICS=busy,temp,power,vcn_activity,jpeg_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none")

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME video-decode
    TARGET videodecode
    GPU ON
    ENVIRONMENT "${_decode_environment}"
    RUN_ARGS -i ${PROJECT_BINARY_DIR}/videos -t 1
    LABELS "decode")

rocprofiler_systems_add_validation_test(
    NAME video-decode-sampling
    PERFETTO_METRIC "rocm_rocdecode_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l rocDecCreateVideoParser -c 2 -d 1 --counter-names "VCN Activity")

# -------------------------------------------------------------------------------------- #
#
# image decode tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME image-decode
    TARGET jpegdecode
    GPU ON
    ENVIRONMENT "${_decode_environment}"
    RUN_ARGS -i ${PROJECT_BINARY_DIR}/images -b 32
    LABELS "decode")

rocprofiler_systems_add_validation_test(
    NAME image-decode-sampling
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l jpegdecode -c 1 -d 0 --counter-names "JPEG Activity")
