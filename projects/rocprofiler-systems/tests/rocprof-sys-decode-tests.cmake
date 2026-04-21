# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# video decode tests
#
# -------------------------------------------------------------------------------------- #

set(_video_decode_environment
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,vcn_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none"
)
set(_jpeg_decode_environment
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocjpeg_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,jpeg_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none"
)

set(_vcn_rocpd_validation_rules
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/video-decode/validation-rules.json"
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/video-decode/sdk-metrics-rules.json"
)

set(_jpeg_rocpd_validation_rules
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/default-rules.json"
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/jpeg-decode/validation-rules.json"
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/jpeg-decode/sdk-metrics-rules.json"
)

# Enable ROCPD for tests only if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _video_decode_environment "ROCPROFSYS_USE_ROCPD=ON")
    list(APPEND _jpeg_decode_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# Engine activity counters are only supported on MI300 and later GPUs
rocprofiler_systems_get_gfx_archs(MI300_DETECTED GFX_MATCH "gfx9[4-9][A-Fa-f0-9]" ECHO)

if(MI300_DETECTED)
    list(APPEND _vcn_counter_names --counter-names "VCN Busy")
    list(APPEND _jpeg_counter_names --counter-names "JPEG Busy")
    list(
        APPEND _vcn_rocpd_validation_rules
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/video-decode/amd-smi-rules.json"
    )
    list(
        APPEND _jpeg_rocpd_validation_rules
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/jpeg-decode/amd-smi-rules.json"
    )
endif()

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME video-decode
    TARGET videodecode
    GPU ON
    ENVIRONMENT "${_base_environment};${_video_decode_environment}"
    RUN_ARGS -i ${PROJECT_BINARY_DIR}/videos -t 1
    LABELS "decode"
)

rocprofiler_systems_add_validation_test(
    NAME video-decode-sampling
    PERFETTO_METRIC "rocm_rocdecode_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l rocDecCreateVideoParser -c 2 -d 1 ${_vcn_counter_names} -p
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU} AND TEST video-decode-sampling)
    set_property(TEST video-decode-sampling APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME video-decode-sampling
        ROCPD_FILE "rocpd.db"
        LABELS "decode;rocpd"
        ARGS --validation-rules
            ${_vcn_rocpd_validation_rules}
    )
endif()

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
    ENVIRONMENT "${_base_environment};${_jpeg_decode_environment}"
    RUN_ARGS -i ${PROJECT_BINARY_DIR}/images -b 32
    LABELS "decode"
)

rocprofiler_systems_add_validation_test(
    NAME jpeg-decode-sampling
    PERFETTO_METRIC "rocm_rocjpeg_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l rocJpegCreate -c 1 -d 1 ${_jpeg_counter_names} -p
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU} AND TEST jpeg-decode-sampling)
    set_property(TEST jpeg-decode-sampling APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME jpeg-decode-sampling
        ROCPD_FILE "rocpd.db"
        LABELS "decode;rocpd"
        ARGS --validation-rules
            ${_jpeg_rocpd_validation_rules}
    )
endif()
