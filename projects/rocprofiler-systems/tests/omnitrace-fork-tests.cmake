# -------------------------------------------------------------------------------------- #
#
# fork tests
#
# -------------------------------------------------------------------------------------- #

omnitrace_add_test(
    NAME fork
    TARGET fork-example
    REWRITE_ARGS -e -v 2 --print-instrumented modules -i 16
    RUNTIME_ARGS -e -v 1 --label file -i 16
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=ON;OMNITRACE_SAMPLING_FREQ=250;OMNITRACE_SAMPLING_REALTIME=ON"
    SAMPLING_PASS_REGEX "fork.. called on PID"
    RUNTIME_PASS_REGEX "fork.. called on PID"
    REWRITE_RUN_PASS_REGEX "fork.. called on PID"
    SAMPLING_FAIL_REGEX
        "(terminate called after throwing an instance|calling abort.. in |Exit code: [1-9])"
    RUNTIME_FAIL_REGEX
        "(terminate called after throwing an instance|calling abort.. in |Exit code: [1-9])"
    REWRITE_RUN_FAIL_REGEX
        "(terminate called after throwing an instance|calling abort.. in |Exit code: [1-9])"
    )
