#
# rocprofv3 tool tests for kfd
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kfd
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

set(kfd-xnack-on-env "${PRELOAD_ENV};HSA_XNACK=1")
set(kfd-xnack-off-env "${PRELOAD_ENV};HSA_XNACK=0")

foreach(_XNACK on off)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-kfd-xnack-${_XNACK}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kfd-trace -d
            ${CMAKE_CURRENT_BINARY_DIR}/kfd-xnack-${_XNACK}-trace -o out --output-format
            json --log-level env -- $<TARGET_FILE:transpose> 1 1 1 2048 2048
        DEPENDS transpose
        TIMEOUT 45
        LABELS "integration-tests;multi-gpu"
        ENVIRONMENT "${kfd-xnack-${_XNACK}-env}"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        FIXTURES_SETUP rocprofv3-test-kfd-xnack-${_XNACK})

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-kfd-xnack-${_XNACK}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        ARGS --json-input
             ${CMAKE_CURRENT_BINARY_DIR}/kfd-xnack-${_XNACK}-trace/out_results.json
        TIMEOUT 45
        LABELS "integration-tests;multi-gpu"
        FAIL_REGULAR_EXPRESSION "AssertionError"
        FIXTURES_REQUIRED rocprofv3-test-kfd-xnack-${_XNACK})
endforeach()
