#
# rocprofv3 tool tests for summaries
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-summary
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Convert the rocpd database to pftrace and otf2 via the rocpd python module
find_package(Python3 REQUIRED)

set(summary-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input-summary.yml.in
               ${CMAKE_CURRENT_BINARY_DIR}/input-summary.yml @ONLY)

##########################################################################################
#
#   Command line input
#
##########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-summary-cmd-line
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace-cmd-line -o out --output-format json rocpd
        --log-level env --runtime-trace --kernel-rename --summary --summary-output-file
        summary --summary-per-domain --summary-groups "KERNEL_DISPATCH|MEMORY_COPY"
        ".*_API" -- $<TARGET_FILE:transpose> 2 500 10
    DEPENDS transpose
    TIMEOUT 45
    WSL_TIMEOUT_SCALE
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}" "ARBITRARY_ENV_VARIABLE=%tag%-trace-cmd-line"
    FIXTURES_SETUP rocprofv3-test-summary-cmd-line)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-summary-cmd-line-rocpd-convert
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace otf2 -d
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line -o out -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${summary-rocpd-env}"
    FIXTURES_REQUIRED rocprofv3-test-summary-cmd-line
    FIXTURES_SETUP rocprofv3-test-summary-cmd-line-rocpd-convert)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-summary-cmd-line
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.pftrace
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.otf2
         --summary-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_summary.txt
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-summary-cmd-line
                      rocprofv3-test-summary-cmd-line-rocpd-convert)

##########################################################################################
#
#   YAML input
#
##########################################################################################

set(summary-yaml-env ${summary-env} "ARBITRARY_ENV_VARIABLE=%tag%-trace-inp-yaml")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-summary-inp-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-summary.yml -- $<TARGET_FILE:transpose> 2 500 10
    DEPENDS transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ARBITRARY_ENV_VARIABLE=%tag%-trace-inp-yaml"
    FIXTURES_SETUP rocprofv3-test-summary-inp-yaml)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-summary-inp-yaml-rocpd-convert
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f otf2 pftrace -d
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml -o out -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${summary-rocpd-env}"
    FIXTURES_REQUIRED rocprofv3-test-summary-inp-yaml
    FIXTURES_SETUP rocprofv3-test-summary-inp-yaml-rocpd-convert)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-summary-inp-yaml
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.pftrace
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.otf2
         --summary-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_summary.txt
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-summary-inp-yaml
                      rocprofv3-test-summary-inp-yaml-rocpd-convert)
