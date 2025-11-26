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

set(spm-dispatch-counter-collection-env
    "${PRELOAD_ENV}" "ROCPROFILER_TOOL_OUTPUT_FILE=spm-dispatch-collection-test.json"
    "ROCPROFILER_TOOL_CONTEXTS=SPM_DISPATCH_COLLECTION"
    "ROCPROFILER_SPM_BETA_ENABLED=True")

rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
set(IS_DISABLED ${IS_SPM_DISABLED})

add_test(NAME test-spm-dispatch-collection-execute COMMAND $<TARGET_FILE:vector-ops>)

set_tests_properties(
    test-spm-dispatch-collection-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${spm-dispatch-counter-collection-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               test-spm-dispatch-collection
               DISABLED
               ${IS_DISABLED})

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(NAME test-spm-dispatch-collection-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/spm-dispatch-collection-test.json)

set_tests_properties(
    test-spm-dispatch-collection-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests"
               DEPENDS
               test-spm-dispatch-collection-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_REQUIRED
               test-spm-dispatch-collection
               DISABLED
               ${IS_DISABLED})
