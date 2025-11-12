# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# -------------------------------------------------------------------------------------- #
#
# jpeg decode tests
#
# -------------------------------------------------------------------------------------- #

set(_jpeg_decode_environment
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocjpeg_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,jpeg_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none"
)

set(_jpeg_rocpd_validation_rules
    "${ROCPROFSYS_ROCPD_VALIDATION_RULES_PATH}/default-rules.json"
    "${ROCPROFSYS_ROCPD_VALIDATION_RULES_PATH}/jpeg-decode/validation-rules.json"
    "${ROCPROFSYS_ROCPD_VALIDATION_RULES_PATH}/jpeg-decode/sdk-metrics-rules.json"
)

# Enable ROCPD for tests only if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _jpeg_decode_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# Engine activity counters are only supported on MI300 and later GPUs
rocprofiler_systems_get_gfx_archs(MI300_DETECTED GFX_MATCH "gfx9[4-9][A-Fa-f0-9]" ECHO)

if(MI300_DETECTED)
    list(APPEND _jpeg_counter_names --counter-names "JPEG Activity")
    list(
        APPEND
        _jpeg_rocpd_validation_rules
        "${ROCPROFSYS_ROCPD_VALIDATION_RULES_PATH}/jpeg-decode/amd-smi-rules.json"
    )
endif()

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME jpeg-decode
    TARGET jpegdecode
    GPU ON
    ENVIRONMENT "${_base_environment};${_jpeg_decode_environment}"
    RUN_ARGS -i ${CMAKE_BINARY_DIR}/images -b 32
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
