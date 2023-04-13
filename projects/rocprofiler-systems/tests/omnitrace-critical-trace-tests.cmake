# -------------------------------------------------------------------------------------- #
#
# critical-trace tests
#
# -------------------------------------------------------------------------------------- #

omnitrace_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_SAMPLING
    NAME parallel-overhead-critical-trace
    TARGET parallel-overhead
    LABELS "critical-trace"
    REWRITE_ARGS
        -e
        -i
        8
        -E
        "^fib"
        -v
        2
        --print-instrumented
        functions
    RUN_ARGS 10 4 100
    ENVIRONMENT "${_critical_trace_environment}")

add_test(
    NAME parallel-overhead-process-critical-trace
    COMMAND
        $<TARGET_FILE:omnitrace-critical-trace>
        ${PROJECT_BINARY_DIR}/omnitrace-tests-output/parallel-overhead-critical-trace-binary-rewrite/call-chain.json
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(_parallel_overhead_critical_trace_environ
    "OMNITRACE_OUTPUT_PATH=omnitrace-tests-output"
    "OMNITRACE_OUTPUT_PREFIX=parallel-overhead-critical-trace/"
    "OMNITRACE_CRITICAL_TRACE_DEBUG=ON"
    "OMNITRACE_VERBOSE=4"
    "OMNITRACE_USE_PID=OFF"
    "OMNITRACE_TIME_OUTPUT=OFF")

set_tests_properties(
    parallel-overhead-process-critical-trace
    PROPERTIES
        ENVIRONMENT
        "${_parallel_overhead_critical_trace_environ}"
        TIMEOUT
        300
        LABELS
        "parallel-overhead;critical-trace"
        PASS_REGULAR_EXPRESSION
        "Outputting.*(critical-trace-cpu.json).*Outputting.*(critical-trace-any.json)"
        DEPENDS
        parallel-overhead-critical-trace-binary-rewrite-run)
