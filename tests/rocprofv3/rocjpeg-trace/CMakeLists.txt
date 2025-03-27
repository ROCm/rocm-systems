#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocjpeg-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocJPEG)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(rocjpeg-tracing-env "${PRELOAD_ENV}")

set(ROCJPEG_IMAGE_DIR "${ROCM_PATH}/share/rocjpeg/images")
if(TARGET rocjpeg-demo AND NOT EXISTS "${ROCJPEG_IMAGE_DIR}")
    message(
        FATAL_ERROR
            "Unable to find image directory for rocjpeg tests: ${ROCJPEG_IMAGE_DIR}")
endif()

# CI Sanitizer run gives the following error: No target "rocjpeg-demo" Adding if-statement
# to avoid tests for now to stop error
add_test(
    NAME rocprofv3-test-rocjpeg-tracing-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --rocjpeg-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json csv
        --log-level env --
        $<IF:$<TARGET_EXISTS:rocjpeg-demo>,$<$<TARGET_EXISTS:rocjpeg-demo>:$<TARGET_FILE:rocjpeg-demo>>,rocjpeg-demo>
        -i ${ROCJPEG_IMAGE_DIR})

set_tests_properties(
    rocprofv3-test-rocjpeg-tracing-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${rocjpeg-tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "threw an exception"
               DISABLED
               $<NOT:$<TARGET_EXISTS:rocjpeg-demo>>)

add_test(
    NAME rocprofv3-test-rocjpeg-tracing-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_results.json --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_results.otf2 --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_results.pftrace --csv-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_rocjpeg_api_trace.csv)

set_tests_properties(
    rocprofv3-test-rocjpeg-tracing-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-rocjpeg-tracing-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               $<NOT:$<TARGET_EXISTS:rocjpeg-demo>>)
