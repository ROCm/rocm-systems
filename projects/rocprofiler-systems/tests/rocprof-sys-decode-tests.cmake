# -------------------------------------------------------------------------------------- #
#
# video decode tests
#
# -------------------------------------------------------------------------------------- #

set(_video_decode_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api"
    "ROCPROFSYS_ROCM_SMI_METRICS=busy,temp,power,vcn_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none")
set(_jpeg_decode_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocjpeg_api"
    "ROCPROFSYS_ROCM_SMI_METRICS=busy,temp,power,jpeg_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none")

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
    ARGS -l
         rocDecCreateVideoParser
         -c
         2
         -d
         1
         --counter-names
         "VCN Activity"
         -p)

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
    ARGS -l
         rocJpegCreate
         -c
         1
         -d
         1
         --counter-names
         "JPEG Activity"
         -p)
