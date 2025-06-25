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
# roctx tests
#
# -------------------------------------------------------------------------------------- #
# Ensure ROCPROFSYS_ROCM_DOMAINS is defined
set(_roctx_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch"
)
rocprofiler_systems_add_test(
    # SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE SKIP_RUNTIME
    NAME roctx-api
    TARGET roctx
    GPU ON
    ENVIRONMENT "${_roctx_environment}"
)
set(ROCTX_LABEL
    roctxMark_GPU_workload
    roctxRangePushA
    roctxRangePushA
    roctxRangeStartA
    roctxRangeStartA
    roctxRangePush_HIP_Kernel
    roctxRangePush_HIP_Kernel
    roctxRangeStart_GPU_Compute
    roctxRangeStart_GPU_Compute
    roctxGetThreadId
    roctxMark_RoctxProfilerPause_End
    roctxMark_Thread_Start
    roctxMark_End
    roctxRangePush_run_profiling
    roctxMark_Finished_GPU
)

set(ROCTX_COUNT
    1
    2
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
)

set(ROCTX_DEPTH
    1
    1
    0
    1
    0
    1
    0
    1
    0
    1
    1
    0
    0
    1
    1
)

rocprofiler_systems_add_validation_test(
    NAME roctx-api-sampling
    PERFETTO_METRIC "rocm_marker_api"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "roctx"
    ARGS -l ${ROCTX_LABEL} -c ${ROCTX_COUNT} -d ${ROCTX_DEPTH} -p
)
