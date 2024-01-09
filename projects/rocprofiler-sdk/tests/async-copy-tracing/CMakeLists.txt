#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-async-copy-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

add_test(NAME test-async-copy-tracing-execute COMMAND $<TARGET_FILE:transpose>)

set_tests_properties(
    test-async-copy-tracing-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>;ROCPROFILER_TOOL_OUTPUT_FILE=async-copy-tracing-test.json"
        FAIL_REGULAR_EXPRESSION
        "threw an exception")

foreach(FILENAME validate.py pytest.ini conftest.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

add_test(NAME test-async-copy-tracing-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/async-copy-tracing-test.json)

set_tests_properties(
    test-async-copy-tracing-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               test-async-copy-tracing-execute FAIL_REGULAR_EXPRESSION
               "threw an exception")
