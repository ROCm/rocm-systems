#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

foreach(FILENAME validate.py input.txt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

# pmc1
add_test(
    NAME rocprofv3-test-counter-collection-pmc1-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -d ${CMAKE_CURRENT_BINARY_DIR}/out_cc_1 -o
        pmc1 $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

set(cc-env-pmc1
    "${PRELOAD_ENV}" "ROCPROF_LOG_LEVEL=${LOG_LEVEL}"
    "ROCPROFILER_LOG_LEVEL=${LOG_LEVEL}"
    "HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>")

set_tests_properties(
    rocprofv3-test-counter-collection-pmc1-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc1}"
               FAIL_REGULAR_EXPRESSION "threw an exception")

add_test(NAME rocprofv3-test-counter-collection-pmc1-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/out_cc_1/pmc_1/pmc1_counter_collection.csv)

set_tests_properties(
    rocprofv3-test-counter-collection-pmc1-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-counter-collection-pmc1-execute FAIL_REGULAR_EXPRESSION
               "threw an exception")
