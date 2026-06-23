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

rocprofiler_add_integration_execute_test(
    rocdecode-tracing
    COMMAND
        $<IF:$<TARGET_EXISTS:rocdecode-demo>,$<TARGET_FILE:rocdecode-demo>,${CMAKE_COMMAND}>
        -i ${ROCDECODE_VIDEO_FILE}
    DEPENDS rocdecode-demo
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED}
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT "ROCPROFILER_TOOL_OUTPUT_FILE=rocdecode-tracing-test.json"
    FIXTURES_SETUP rocdecode-tracing)

rocprofiler_add_integration_validate_test(
    rocdecode-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    FIXTURES_REQUIRED rocdecode-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/rocdecode-tracing-test.json)
