# -------------------------------------------------------------------------------------- #
#
# binary-rewrite and runtime-instrumentation tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_SAMPLING SKIP_RUNTIME
    NAME rewrite-caller
    TARGET rewrite-caller
    LABELS "caller-include"
    REWRITE_ARGS
        -e
        -i
        256
        --caller-include
        "^inner"
        -v
        2
        --print-instrumented
        functions
    RUN_ARGS 17
    ENVIRONMENT "${_base_environment};ROCPROFSYS_COUT_OUTPUT=ON"
    BASELINE_PASS_REGEX "number of calls made = 17"
    REWRITE_PASS_REGEX "\\[function\\]\\[Forcing\\] caller-include-regex :: 'outer'"
    REWRITE_RUN_PASS_REGEX ">>> ._outer ([ \\|]+) 17")

rocprofiler_systems_add_test(
    NAME parallel-overhead
    TARGET parallel-overhead
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=8
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}")

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME parallel-overhead-locks-perfetto
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUN_ARGS 10 4 1000
    ENVIRONMENT
        "${_lock_environment};ROCPROFSYS_FLAT_PROFILE=ON;ROCPROFSYS_PROFILE=OFF;ROCPROFSYS_TRACE=ON;ROCPROFSYS_SAMPLING_KEEP_INTERNAL=OFF"
    )
