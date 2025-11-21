#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocdecode-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocDecode)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(rocdecode-tracing-env "${PRELOAD_ENV}")

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

# rocDecode tracing test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocdecode-tracing
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --rocdecode-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json csv pftrace
        otf2 --log-level env --
        $<IF:$<TARGET_EXISTS:rocdecode-demo>,$<TARGET_FILE:rocdecode-demo>,rocdecode-demo>
        -i ${ROCDECODE_VIDEO_FILE}
    DEPENDS rocdecode-demo
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-rocdecode-tracing
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-rocdecode-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocdecode-demo-trace/out_results.json
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocdecode-demo-trace/out_results.otf2
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocdecode-demo-trace/out_results.pftrace
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocdecode-demo-trace/out_rocdecode_api_trace.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-rocdecode-tracing
    DISABLED "${IS_DISABLED}")
