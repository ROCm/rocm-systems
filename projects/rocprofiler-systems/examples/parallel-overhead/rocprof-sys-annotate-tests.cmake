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
# papi tests
#
# -------------------------------------------------------------------------------------- #

if(
    ROCPROFSYS_USE_PAPI
    AND (
        rocprofiler_systems_perf_event_paranoid LESS_EQUAL 3
        OR rocprofiler_systems_cap_sys_admin EQUAL 0
        OR rocprofiler_systems_cap_perfmon EQUAL 0
    )
)
    set(_annotate_environment
        "${_base_environment}"
        "ROCPROFSYS_TIMEMORY_COMPONENTS=thread_cpu_clock papi_array"
        "ROCPROFSYS_PAPI_EVENTS=perf::PERF_COUNT_SW_CPU_CLOCK"
        "ROCPROFSYS_USE_SAMPLING=OFF"
    )

    rocprofiler_systems_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME annotate
        TARGET parallel-overhead
        RUN_ARGS 30 2 200
        REWRITE_ARGS
            -e
            -v
            2
            -R
            run
            --allow-overlapping
            --print-available
            functions
            --print-overlapping
            functions
            --print-excluded
            functions
            --print-instrumented
            functions
            --print-instructions
        ENVIRONMENT "${_annotate_environment}"
        LABELS "annotate;papi"
    )

    rocprofiler_systems_add_validation_test(
        NAME annotate-binary-rewrite
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "annotate;papi"
        ARGS --key-names perf::PERF_COUNT_SW_CPU_CLOCK thread_cpu_clock --key-counts 8 8
    )

    rocprofiler_systems_add_validation_test(
        NAME annotate-sampling
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "papi"
        ARGS --key-names thread_cpu_clock --key-counts 6
    )
else()
    set(_annotate_environment
        "${_base_environment}"
        "ROCPROFSYS_TIMEMORY_COMPONENTS=thread_cpu_clock"
        "ROCPROFSYS_USE_SAMPLING=OFF"
    )

    rocprofiler_systems_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME annotate
        TARGET parallel-overhead
        RUN_ARGS 30 2 200
        REWRITE_ARGS
            -e
            -v
            2
            -R
            run
            --allow-overlapping
            --print-available
            functions
            --print-overlapping
            functions
            --print-excluded
            functions
            --print-instrumented
            functions
            --print-instructions
        ENVIRONMENT "${_annotate_environment}"
        LABELS "annotate"
    )

    rocprofiler_systems_add_validation_test(
        NAME annotate-binary-rewrite
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "annotate"
        ARGS --key-names thread_cpu_clock --key-counts 8
    )

    rocprofiler_systems_add_validation_test(
        NAME annotate-sampling
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "annotate"
        ARGS --key-names thread_cpu_clock --key-counts 6
    )
endif()
