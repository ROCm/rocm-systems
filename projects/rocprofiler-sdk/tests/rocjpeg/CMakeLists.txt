#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocjpeg-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocJPEG)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")
set(rocjpeg-tracing-env "${PRELOAD_ENV}")

set(rocJPEG_IMAGE_DIR "${rocJPEG_ROOT_DIR}/share/rocjpeg/images")
if(TARGET rocjpeg-demo AND NOT EXISTS "${rocJPEG_IMAGE_DIR}")
    message(
        FATAL_ERROR
            "Unable to find image directory for rocjpeg tests: ${rocJPEG_IMAGE_DIR}")
endif()

add_test(
    NAME test-rocjpeg-tracing-execute
    COMMAND
        $<IF:$<TARGET_EXISTS:rocjpeg-demo>,$<$<TARGET_EXISTS:rocjpeg-demo>:$<TARGET_FILE:rocjpeg-demo>>,${CMAKE_COMMAND}>
        -i ${rocJPEG_IMAGE_DIR})

set(rocjpeg-tracing-env
    "${PRELOAD_ENV}"
    "ROCPROFILER_TOOL_OUTPUT_FILE=rocjpeg-tracing-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-rocjpeg-tracing-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${rocjpeg-tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               $<NOT:$<TARGET_EXISTS:rocjpeg-demo>>)

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(NAME test-rocjpeg-tracing-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-tracing-test.json)

set_tests_properties(
    test-rocjpeg-tracing-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               test-rocjpeg-tracing-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               $<NOT:$<TARGET_EXISTS:rocjpeg-demo>>)
