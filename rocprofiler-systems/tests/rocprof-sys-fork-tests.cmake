# -------------------------------------------------------------------------------------- #
#
# fork tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    NAME fork
    TARGET fork-example
    REWRITE_ARGS -e -v 2 --print-instrumented modules -i 16
    RUNTIME_ARGS -e -v 1 --label file -i 16
    ENVIRONMENT
        "${_base_environment};ROCPROFSYS_SAMPLING_FREQ=250;ROCPROFSYS_SAMPLING_REALTIME=ON"
    SAMPLING_PASS_REGEX "fork.. called on PID"
    RUNTIME_PASS_REGEX "fork.. called on PID"
    REWRITE_RUN_PASS_REGEX "fork.. called on PID"
    SAMPLING_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})"
    RUNTIME_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})"
    REWRITE_RUN_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
