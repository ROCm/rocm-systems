# -------------------------------------------------------------------------------------- #
#
# ROCm tests
#
# -------------------------------------------------------------------------------------- #

set(ROCPROFSYS_ROCM_EVENTS_TEST "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0")

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
    RUNTIME_TIMEOUT 480)

rocprofiler_systems_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME transpose-two-kernels
    TARGET transpose
    MPI OFF
    GPU ON
    NUM_PROCS 1
    RUN_ARGS 1 2 2
    ENVIRONMENT "${_base_environment}")

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
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure transpose")

if(ROCPROFSYS_USE_ROCM)
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
        SAMPLING_PASS_REGEX "${_ROCP_PASS_REGEX}")

    rocprofiler_systems_add_validation_test(
        NAME transpose-rocprofiler-sampling
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS --counter-names "TA_TA_BUSY" "SQ_WAVES" "GRBM_COUNT" "SQ_INSTS_VALU" -p
        EXIST_FILES rocprof-device-0-GRBM_COUNT.txt rocprof-device-0-TA_TA_BUSY.txt
                    rocprof-device-0-SQ_INSTS_VALU.txt rocprof-device-0-SQ_WAVES.txt
        LABELS "rocprofiler")

    rocprofiler_systems_add_validation_test(
        NAME transpose-rocprofiler-binary-rewrite
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS --counter-names "TA_TA_BUSY" "SQ_WAVES" "GRBM_COUNT" "SQ_INSTS_VALU" -p
        EXIST_FILES rocprof-device-0-GRBM_COUNT.txt rocprof-device-0-TA_TA_BUSY.txt
                    rocprof-device-0-SQ_INSTS_VALU.txt rocprof-device-0-SQ_WAVES.txt
        LABELS "rocprofiler")
endif()
