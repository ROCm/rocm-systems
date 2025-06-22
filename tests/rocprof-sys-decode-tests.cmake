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
# video decode tests
#
# -------------------------------------------------------------------------------------- #

set(_video_decode_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,vcn_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none"
)
set(_jpeg_decode_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocjpeg_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,jpeg_activity,mem_usage"
    "ROCPROFSYS_SAMPLING_CPUS=none"
)

rocprofiler_systems_get_gfx_archs(MI300_DETECTED GFX_MATCH "gfx9[4-9][A-Fa-f0-9]" ECHO)

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
    LABELS "decode"
)

rocprofiler_systems_add_validation_test(
    NAME video-decode-sampling
    PERFETTO_METRIC "rocm_rocdecode_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l rocDecCreateVideoParser -c 2 -d 1 ${VCN_COUNTER_NAMES_ARG} -p
)

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
    LABELS "decode"
)

rocprofiler_systems_add_validation_test(
    NAME jpeg-decode-sampling
    PERFETTO_METRIC "rocm_rocjpeg_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "decode"
    ARGS -l rocJpegCreate -c 1 -d 1 ${JPEG_COUNTER_NAMES_ARG} -p
)
