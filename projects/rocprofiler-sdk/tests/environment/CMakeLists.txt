#
# Environment variable tests
#
# Tests that environment variables are correctly read across different scenarios:
# - bash as target (bypasses bash's custom getenv())
# - Other shells and interpreters
# - Various environment configurations
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-environment
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Note: bash is invoked unconditionally (no find_program guard) because these tests are
# the regression coverage for the bash-specific getenv() bug — silently skipping them when
# bash is absent would hide the bug they exist to catch.

# Test 1: Invalid ROCPROFILER_METRICS_PATH with bash should show warning This verifies
# that the environment variable is actually read (not ignored).
rocprofiler_add_integration_execute_test(
    environment-bash-invalid-path
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES -- bash -c "exit 0"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 30
    LABELS "integration-tests" "environment"
    ENVIRONMENT "ROCPROFILER_METRICS_PATH=/nonexistent/bash/test/path"
    # Expect warning about missing config file - proves env var was read
    PASS_REGULAR_EXPRESSION
        "Counter definitions file '/nonexistent/bash/test/path/config.yaml' does not exist"
    FIXTURES_SETUP environment-bash-invalid-path)

# Test 2: Bash with path containing spaces Validates that paths with special characters
# are handled correctly
rocprofiler_add_integration_execute_test(
    environment-bash-path-with-spaces
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES -- bash -c "exit 0"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 30
    LABELS "integration-tests" "environment" "bash"
    ENVIRONMENT "ROCPROFILER_METRICS_PATH=/path/with spaces/metrics"
    # Should see the path with spaces in the error message
    PASS_REGULAR_EXPRESSION "Counter definitions file '/path/with spaces/metrics"
    FIXTURES_SETUP environment-bash-spaces)

# Test 3: Multiple environment variables under bash Validates that multiple ROCPROFILER_*
# env vars are all readable
rocprofiler_add_integration_execute_test(
    environment-bash-multiple-vars
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES -- bash -c "exit 0"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 30
    LABELS "integration-tests" "environment" "bash"
    ENVIRONMENT "ROCPROFILER_METRICS_PATH=/test/path/one" "ROCPROFILER_LOG_LEVEL=trace"
    # Should see both env vars being used
    PASS_REGULAR_EXPRESSION "Counter definitions file '/test/path/one"
    FIXTURES_SETUP environment-bash-multiple)

# Test 4: Invalid ROCPROFILER_METRICS_PATH with sh should show warning
find_program(SH_EXECUTABLE sh)
if(SH_EXECUTABLE)
    rocprofiler_add_integration_execute_test(
        environment-sh-invalid-path
        COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --
                ${SH_EXECUTABLE} -c "exit 0"
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 30
        LABELS "integration-tests" "environment" "sh"
        ENVIRONMENT "ROCPROFILER_METRICS_PATH=/nonexistent/sh/test"
        PASS_REGULAR_EXPRESSION "Counter definitions file '/nonexistent/sh/test"
        FIXTURES_SETUP environment-sh-test)
endif()

# Test 5: Invalid ROCPROFILER_METRICS_PATH with zsh should show warning
find_program(ZSH_EXECUTABLE zsh)
if(ZSH_EXECUTABLE)
    rocprofiler_add_integration_execute_test(
        environment-zsh-invalid-path
        COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --
                ${ZSH_EXECUTABLE} -c "exit 0"
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 30
        LABELS "integration-tests" "environment" "zsh"
        ENVIRONMENT "ROCPROFILER_METRICS_PATH=/nonexistent/zsh/test"
        PASS_REGULAR_EXPRESSION "Counter definitions file '/nonexistent/zsh/test"
        FIXTURES_SETUP environment-zsh-test)
endif()

# Test 6: Bash with inline script (mimics user wrapper scripts) Real-world scenario: users
# often wrap profiling commands in bash -c This test also validates normal operation (no
# spurious warnings)
rocprofiler_add_integration_execute_test(
    environment-bash-inline-script
    COMMAND
        bash -c
        "export ROCPROFILER_METRICS_PATH=/inline/test && \
                     $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES -- bash -c 'exit 0'"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 30
    LABELS "integration-tests" "environment" "bash" "real-world"
    # Should see the inline-set env var
    PASS_REGULAR_EXPRESSION "Counter definitions file '/inline/test"
    FIXTURES_SETUP environment-bash-inline)
