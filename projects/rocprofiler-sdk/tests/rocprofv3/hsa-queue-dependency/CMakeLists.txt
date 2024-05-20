#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-hsa-queue-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "${PRELOAD_ENV}")

foreach(FILENAME validate.py conftest.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

find_package(rocprofiler-sdk REQUIRED)

# hsa multiqueue dependency test
add_test(
    NAME rocprofv3-test-hsa-multiqueue-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out $<TARGET_FILE:multiqueue_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-execute
    PROPERTIES LABELS "integration-tests" ENVIRONMENT "${tracing-env}"
               FAIL_REGULAR_EXPRESSION "HSA_API|HIP_API")

add_test(
    NAME rocprofv3-test-hsa-multiqueue-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --hsa-trace-input
        ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_hsa_api_trace.csv
        --kernel-trace-input
        ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_kernel_trace.csv)

set(MULTIQUEUE_VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_hsa_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_kernel_api_trace.csv)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-hsa-multiqueue-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               ATTACHED_FILES_ON_FAIL
               "${MULTIQUEUE_VALIDATION_FILES}")
