#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-memory-allocation-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

add_test(NAME test-memory-allocation-tracing-execute
         COMMAND $<TARGET_FILE:hsa-memory-allocation>)

set(memory-allocation-tracing-env
    "${PRELOAD_ENV}"
    "ROCPROFILER_TOOL_OUTPUT_FILE=memory-allocation-tracing-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-memory-allocation-tracing-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               "${memory-allocation-tracing-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(NAME test-memory-allocation-tracing-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/memory-allocation-tracing-test.json)

set_tests_properties(
    test-memory-allocation-tracing-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               test-memory-allocation-tracing-execute FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
