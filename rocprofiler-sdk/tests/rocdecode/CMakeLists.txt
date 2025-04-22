#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocdecode-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocDecode)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

file(GLOB_RECURSE ROCDECODE_VIDEO_FILE
     "${rocDecode_ROOT_DIR}/share/rocdecode/video/*AMD_driving_virtual_20-H265.265")
if(TARGET rocdecode-demo AND NOT EXISTS "${ROCDECODE_VIDEO_FILE}")
    message(
        WARNING "Unable to find video file for rocdecode tests: ${ROCDECODE_VIDEO_FILE}")
    set(IS_DISABLED ON)
endif()

if(NOT TARGET rocdecode-demo)
    set(IS_DISABLED ON)
endif()

add_test(
    NAME test-rocdecode-tracing-execute
    COMMAND
        $<IF:$<TARGET_EXISTS:rocdecode-demo>,$<$<TARGET_EXISTS:rocdecode-demo>:$<TARGET_FILE:rocdecode-demo>>,${CMAKE_COMMAND}>
        -i ${ROCDECODE_VIDEO_FILE})

set(rocdecode-tracing-env
    "${PRELOAD_ENV}"
    "ROCPROFILER_TOOL_OUTPUT_FILE=rocdecode-tracing-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-rocdecode-tracing-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${rocdecode-tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               $<NOT:$<TARGET_EXISTS:rocdecode-demo>>)

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(NAME test-rocdecode-tracing-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/rocdecode-tracing-test.json)

set_tests_properties(
    test-rocdecode-tracing-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               test-rocdecode-tracing-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               $<NOT:$<TARGET_EXISTS:rocdecode-demo>>)
