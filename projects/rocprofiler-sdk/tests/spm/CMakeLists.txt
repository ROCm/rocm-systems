#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-spm-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

if(NOT ROCPROFILER_MEMCHECK STREQUAL "")
    set(SANITIZER True)
else()
    set(SANITIZER False)
endif()

rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
if(${SANITIZER})
    set(IS_SPM_DISABLED True)
endif()
set(IS_DISABLED True)

rocprofiler_add_integration_execute_test(
    test-spm-dispatch-collection-execute
    TARGET simple-transpose
    TIMEOUT 60
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=spm-dispatch-collection-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=SPM_DISPATCH_COLLECTION"
        "ROCPROFILER_SPM_BETA_ENABLED=True"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP spm-dispatch
    DISABLED ${IS_DISABLED})

rocprofiler_add_integration_execute_test(
    test-spm-dispatch-buffer-collection-execute
    TARGET simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=spm-buffer-dispatch-collection-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=SPM_BUFFER_DISPATCH_COLLECTION"
        "ROCPROFILER_SPM_BETA_ENABLED=True"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP spm-buffer
    DISABLED ${IS_DISABLED})

rocprofiler_add_integration_validate_test(
    test-spm-dispatch-collection-validate
    TEST_PATHS validate.py
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/spm-dispatch-collection-test.json
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 160
    LABELS "integration-tests"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED spm-dispatch
    DISABLED ${IS_DISABLED})

rocprofiler_add_integration_validate_test(
    test-spm-dispatch-buffer-collection-validate
    TEST_PATHS validate.py
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/spm-buffer-dispatch-collection-test.json
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 160
    LABELS "integration-tests"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED spm-buffer
    DISABLED ${IS_DISABLED})
