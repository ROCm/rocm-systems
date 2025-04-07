# -------------------------------------------------------------------------------------- #
#
# video decode tests
#
# -------------------------------------------------------------------------------------- #

set(_video_decode_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,vcn_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none")
set(_jpeg_decode_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocjpeg_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,jpeg_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none")

check_gpu("MI300" MI300_DETECTED)
if(MI300_DETECTED)
    list(APPEND VCN_COUNTER_NAMES_ARG --counter-names "VCN Activity")
    list(APPEND JPEG_COUNTER_NAMES_ARG --counter-names "JPEG Activity")
endif()

# check_gpu("MI100" MI100_DETECTED) if(MI100_DETECTED) list(APPEND VCN_COUNTER_NAMES_ARG
# --counter-names "VCN Activity") endif()

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME video-decode
    TARGET videodecode
    GPU ON
    ENVIRONMENT "${_video_decode_environment}"
    RUN_ARGS -i ${PROJECT_BINARY_DIR}/videos -t 1
    LABELS "decode")

rocprofiler_systems_add_validation_test(
    NAME video-decode-sampling
    PERFETTO_METRIC "rocm_rocdecode_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l rocDecCreateVideoParser -c 2 -d 1 ${VCN_COUNTER_NAMES_ARG} -p)

# -------------------------------------------------------------------------------------- #
#
# jpeg decode tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME jpeg-decode
    TARGET jpegdecode
    GPU ON
    ENVIRONMENT "${_jpeg_decode_environment}"
    RUN_ARGS -i ${PROJECT_BINARY_DIR}/images -b 32
    LABELS "decode")

rocprofiler_systems_add_validation_test(
    NAME jpeg-decode-sampling
    PERFETTO_METRIC "rocm_rocjpeg_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l rocJpegCreate -c 1 -d 1 ${JPEG_COUNTER_NAMES_ARG} -p)
