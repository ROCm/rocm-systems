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
# ROCm tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    NAME transpose
    TARGET transpose
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 2 --print-instructions -E uniform_int_distribution
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -E
        uniform_int_distribution
    ENVIRONMENT "${_base_environment}"
    RUNTIME_TIMEOUT 480
)

rocprofiler_systems_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME transpose-two-kernels
    TARGET transpose
    MPI OFF
    GPU ON
    NUM_PROCS 1
    RUN_ARGS 1 2 2
    ENVIRONMENT "${_base_environment}"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME transpose-loops
    TARGET transpose
    LABELS "loops"
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS
        -e
        -v
        2
        --label
        return
        args
        -l
        -i
        8
        -E
        uniform_int_distribution
    RUN_ARGS 2 100 50
    ENVIRONMENT "${_base_environment}"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure transpose"
)

if(ROCPROFSYS_USE_ROCM)
    set(NAVI_REGEX "gfx(10|11|12)[A-Fa-f0-9][A-Fa-f0-9]")
    rocprofiler_systems_get_gfx_archs(NAVI_DETECTED GFX_MATCH ${NAVI_REGEX} ECHO)

    if(NAVI_DETECTED)
        set(ROCPROFSYS_ROCM_EVENTS_TEST "SQ_WAVES")
        set(ROCPROFSYS_FILE_CHECKS "rocprof-device-0-SQ_WAVES.txt")
        set(ROCPROFSYS_COUNTER_NAMES_ARG "SQ_WAVES")
    else()
        set(ROCPROFSYS_ROCM_EVENTS_TEST
            "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0"
        )
        set(ROCPROFSYS_FILE_CHECKS
            "rocprof-device-0-GRBM_COUNT.txt"
            "rocprof-device-0-SQ_WAVES.txt"
            "rocprof-device-0-SQ_INSTS_VALU.txt"
            "rocprof-device-0-TA_TA_BUSY.txt"
        )
        set(ROCPROFSYS_COUNTER_NAMES_ARG
            "GRBM_COUNT"
            "SQ_WAVES"
            "SQ_INSTS_VALU"
            "TA_TA_BUSY"
        )
    endif()

    rocprofiler_systems_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME transpose-rocprofiler
        TARGET transpose
        LABELS "rocprofiler"
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        REWRITE_ARGS -e -v 2 -E uniform_int_distribution
        ENVIRONMENT
            "${_base_environment};ROCPROFSYS_ROCM_EVENTS=${ROCPROFSYS_ROCM_EVENTS_TEST}"
        REWRITE_RUN_PASS_REGEX "${_ROCP_PASS_REGEX}"
        SAMPLING_PASS_REGEX "${_ROCP_PASS_REGEX}"
    )

    rocprofiler_systems_add_validation_test(
        NAME transpose-rocprofiler-sampling
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS --counter-names ${ROCPROFSYS_COUNTER_NAMES_ARG} -p
        EXIST_FILES ${ROCPROFSYS_FILE_CHECKS}
        LABELS "rocprofiler"
    )

    rocprofiler_systems_add_validation_test(
        NAME transpose-rocprofiler-binary-rewrite
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS --counter-names ${ROCPROFSYS_COUNTER_NAMES_ARG} -p
        EXIST_FILES ${ROCPROFSYS_FILE_CHECKS}
        LABELS "rocprofiler"
    )
endif()
