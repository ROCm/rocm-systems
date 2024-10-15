# -------------------------------------------------------------------------------------- #
#
# code-coverage tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%")

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%")

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-basic-blocks
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%")

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-basic-blocks-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%")
