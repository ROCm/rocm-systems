#
# rocprofv3 tool
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

foreach(FILENAME validate.py conftest.py pytest.ini input.txt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

# pmc3
add_test(
    NAME rocprofv3-test-tracing-plus-cc-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -d ${CMAKE_CURRENT_BINARY_DIR}/out_cc_trace
        -o pmc3 $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

set(cc-tracing-env
    "${PRELOAD_ENV}" "ROCPROF_LOG_LEVEL=${LOG_LEVEL}"
    "ROCPROFILER_LOG_LEVEL=${LOG_LEVEL}"
    "HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>")

set_tests_properties(
    rocprofv3-test-tracing-plus-cc-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-tracing-env}"
               FAIL_REGULAR_EXPRESSION "threw an exception")

add_test(NAME rocprofv3-test-tracing-plus-cc-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py
                 --input-dir "${CMAKE_CURRENT_BINARY_DIR}/out_cc_trace")

set_tests_properties(
    rocprofv3-test-tracing-plus-cc-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-tracing-plus-cc-execute FAIL_REGULAR_EXPRESSION
               "threw an exception")
