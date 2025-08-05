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
# overflow tests
#
# -------------------------------------------------------------------------------------- #

set(_overflow_environment
    "${_base_environment}"
    "ROCPROFSYS_VERBOSE=2"
    "ROCPROFSYS_SAMPLING_CPUTIME=OFF"
    "ROCPROFSYS_SAMPLING_REALTIME=OFF"
    "ROCPROFSYS_SAMPLING_OVERFLOW=ON"
    "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=PERF_COUNT_SW_CPU_CLOCK"
    "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ=10000"
    "ROCPROFSYS_DEBUG_THREADING_GET_ID=ON"
)

if(
    rocprofiler_systems_perf_event_paranoid LESS_EQUAL 3
    OR rocprofiler_systems_cap_sys_admin EQUAL 0
    OR rocprofiler_systems_cap_perfmon EQUAL 0
)
    rocprofiler_systems_add_test(
        SKIP_BASELINE
        NAME overflow
        TARGET parallel-overhead
        RUN_ARGS 30 2 200
        REWRITE_ARGS -e -v 2
        RUNTIME_ARGS -e -v 1
        ENVIRONMENT "${_overflow_environment}"
        LABELS "perf;overflow"
        SAMPLING_PASS_REGEX "sampling_wall_clock.txt"
        RUNTIME_PASS_REGEX "sampling_wall_clock.txt"
        REWRITE_RUN_PASS_REGEX "sampling_wall_clock.txt"
    )
endif()
