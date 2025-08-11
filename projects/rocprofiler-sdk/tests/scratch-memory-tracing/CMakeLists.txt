#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-scratch-memory-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

add_test(NAME test-scratch-memory-tracing-execute COMMAND $<TARGET_FILE:scratch-memory>)

set(scratch-memory-tracing-env
    "${PRELOAD_ENV}"
    "ROCPROFILER_TOOL_OUTPUT_FILE=scratch-memory-tracing-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-scratch-memory-tracing-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${scratch-memory-tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}"
               FIXTURES_SETUP
               scratch-memory-trace)

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(NAME test-scratch-memory-tracing-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/scratch-memory-tracing-test.json)

set_tests_properties(
    test-scratch-memory-tracing-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               test-scratch-memory-tracing-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}"
               FIXTURES_REQUIRED
               scratch-memory-trace)
