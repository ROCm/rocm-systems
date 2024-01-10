# -------------------------------------------------------------------------------------- #
#
# ROCm tests
#
# -------------------------------------------------------------------------------------- #

set(OMNITRACE_ROCM_EVENTS_TEST
    "GRBM_COUNT,GPUBusy,SQ_WAVES,SQ_INSTS_VALU,VALUInsts,TCC_HIT_sum,TA_TA_BUSY[0]:device=0,TA_TA_BUSY[11]:device=0"
    )

omnitrace_add_test(
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
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=ON")

omnitrace_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME transpose-two-kernels
    TARGET transpose
    MPI OFF
    GPU ON
    NUM_PROCS 1
    RUN_ARGS 1 2 2
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_ROCTRACER_HSA_ACTIVITY=OFF;OMNITRACE_ROCTRACER_HSA_API=OFF"
    )

omnitrace_add_test(
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
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure transpose")

if(OMNITRACE_USE_ROCPROFILER)
    omnitrace_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME transpose-rocprofiler
        TARGET transpose
        LABELS "rocprofiler"
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        REWRITE_ARGS -e -v 2 -E uniform_int_distribution
        ENVIRONMENT
            "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_ROCM_EVENTS=${OMNITRACE_ROCM_EVENTS_TEST}"
        REWRITE_RUN_PASS_REGEX
            "rocprof-device-0-GRBM_COUNT.txt(.*)rocprof-device-0-GPUBusy.txt(.*)rocprof-device-0-SQ_WAVES.txt(.*)rocprof-device-0-SQ_INSTS_VALU.txt(.*)rocprof-device-0-VALUInsts.txt(.*)rocprof-device-0-TCC_HIT_sum.txt(.*)rocprof-device-0-TA_TA_BUSY_0.txt(.*)rocprof-device-0-TA_TA_BUSY_11.txt"
        )

    omnitrace_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME transpose-rocprofiler-no-roctracer
        TARGET transpose
        LABELS "rocprofiler"
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        REWRITE_ARGS -e -v 2 -E uniform_int_distribution
        ENVIRONMENT
            "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_ROCTRACER=OFF;OMNITRACE_ROCM_EVENTS=${OMNITRACE_ROCM_EVENTS_TEST}"
        REWRITE_RUN_PASS_REGEX
            "rocprof-device-0-GRBM_COUNT.txt(.*)rocprof-device-0-GPUBusy.txt(.*)rocprof-device-0-SQ_WAVES.txt(.*)rocprof-device-0-SQ_INSTS_VALU.txt(.*)rocprof-device-0-VALUInsts.txt(.*)rocprof-device-0-TCC_HIT_sum.txt(.*)rocprof-device-0-TA_TA_BUSY_0.txt(.*)rocprof-device-0-TA_TA_BUSY_11.txt"
        REWRITE_RUN_FAIL_REGEX "roctracer.txt|OMNITRACE_ABORT_FAIL_REGEX")
endif()
