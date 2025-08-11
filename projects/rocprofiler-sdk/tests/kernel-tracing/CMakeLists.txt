#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-kernel-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(PYTEST_ARGS)
if(ROCPROFILER_MEMCHECK MATCHES "(Address|Thread|UndefinedBehavior)Sanitizer"
   OR ROCPROFILER_BUILD_CODECOV
   OR ROCPROFILER_DISABLE_UNSTABLE_CTESTS)
    set(PYTEST_ARGS -k "not test_total_runtime")
endif()

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

add_test(NAME test-kernel-tracing-execute COMMAND $<TARGET_FILE:reproducible-runtime>)

set(kernel-tracing-env
    "${PRELOAD_ENV}"
    "ROCPROFILER_TOOL_OUTPUT_FILE=kernel-tracing-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-kernel-tracing-execute
    PROPERTIES TIMEOUT
               100
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${kernel-tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               kernel-tracing)

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(
    NAME test-kernel-tracing-validate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py ${PYTEST_ARGS}
            --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test.json)

set_tests_properties(
    test-kernel-tracing-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               test-kernel-tracing-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_REQUIRED
               kernel-tracing)
