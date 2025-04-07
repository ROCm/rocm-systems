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
    "ROCPROFSYS_DEBUG_THREADING_GET_ID=ON")

if(rocprofiler_systems_perf_event_paranoid LESS_EQUAL 3
   OR rocprofiler_systems_cap_sys_admin EQUAL 0
   OR rocprofiler_systems_cap_perfmon EQUAL 0)
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
        REWRITE_RUN_PASS_REGEX "sampling_wall_clock.txt")
endif()
