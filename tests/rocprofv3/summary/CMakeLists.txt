#
# rocprofv3 tool tests for summaries
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-summary
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(summary-env "${PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini input-summary.yaml COPY validate.py
                                                                             conftest.py)

##########################################################################################
#
#   Command line input
#
##########################################################################################

set(summary-cmdl-env ${summary-env} "ARBITRARY_ENV_VARIABLE=%tag%-trace-cmd-line")

add_test(
    NAME rocprofv3-test-summary-cmd-line-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace-cmd-line -o out --output-format pftrace
        json otf2 --log-level env --runtime-trace --kernel-rename --summary
        --summary-output-file summary --summary-per-domain --summary-groups
        "KERNEL_DISPATCH|MEMORY_COPY" ".*_API" -- $<TARGET_FILE:transpose> 2 500 10)

set_tests_properties(
    rocprofv3-test-summary-cmd-line-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${summary-cmdl-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-summary-cmd-line-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.pftrace
        --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.otf2
        --summary-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_summary.txt)

set_tests_properties(
    rocprofv3-test-summary-cmd-line-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-summary-execute" FAIL_REGULAR_EXPRESSION "AssertionError")

##########################################################################################
#
#   YAML input
#
##########################################################################################

set(summary-yaml-env ${summary-env} "ARBITRARY_ENV_VARIABLE=%tag%-trace-inp-yaml")

add_test(
    NAME rocprofv3-test-summary-inp-yaml-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-summary.yaml -- $<TARGET_FILE:transpose> 2 500
        10)

set_tests_properties(
    rocprofv3-test-summary-inp-yaml-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${summary-yaml-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-summary-inp-yaml-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.pftrace
        --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.otf2
        --summary-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_summary.txt)

set_tests_properties(
    rocprofv3-test-summary-inp-yaml-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-summary-inp-yaml-execute" FAIL_REGULAR_EXPRESSION
               "AssertionError")
