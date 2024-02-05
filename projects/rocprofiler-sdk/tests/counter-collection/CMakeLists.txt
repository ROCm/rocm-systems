#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

add_test(NAME test-counter-collection-execute COMMAND $<TARGET_FILE:multistream>)

set_tests_properties(
    test-counter-collection-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>;ROCPROFILER_TOOL_OUTPUT_FILE=counter-collection-test.json;ROCPROFILER_TOOL_CONTEXTS=COUNTER_COLLECTION;ROCPROF_COUNTERS=SQ_WAVES_sum"
        FAIL_REGULAR_EXPRESSION
        "threw an exception")

foreach(FILENAME validate.py pytest.ini conftest.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

add_test(NAME test-counter-collection-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/counter-collection-test.json)

set_tests_properties(
    test-counter-collection-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               test-counter-collection-execute FAIL_REGULAR_EXPRESSION
               "threw an exception")
