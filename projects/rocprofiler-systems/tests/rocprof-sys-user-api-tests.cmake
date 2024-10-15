# -------------------------------------------------------------------------------------- #
#
# User API tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    NAME user-api
    TARGET user-api
    LABELS "loops"
    REWRITE_ARGS -e -v 2 -l --min-instructions=8 -E custom_push_region
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --min-instructions=8
        -E
        custom_push_region
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    REWRITE_RUN_PASS_REGEX "Pushing custom region :: run.10. x 1000"
    RUNTIME_PASS_REGEX "Pushing custom region :: run.10. x 1000"
    SAMPLING_PASS_REGEX "Pushing custom region :: run.10. x 1000"
    BASELINE_FAIL_REGEX "Pushing custom region"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")
