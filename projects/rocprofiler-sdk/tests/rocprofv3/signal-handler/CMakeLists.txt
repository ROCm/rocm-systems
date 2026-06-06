#
# rocprofv3 signal handler tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-signal-handler
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# Signal delay: how long the app runs before timeout sends SIGINT
set(SIGNAL_DELAY 3)
set(CTEST_TIMEOUT 30)

# timeout sends SIGINT after SIGNAL_DELAY seconds (simulates Ctrl+C). --preserve-status
# returns the child's actual exit code. If our signal handling hangs, the test fails via
# ctest TIMEOUT ${CTEST_TIMEOUT}
set(TIMEOUT_CMD timeout --preserve-status --signal=INT ${SIGNAL_DELAY})

# Output directory base
set(BASE_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Process types to test
set(PROCESS_TYPES single-process fork fork-exec spawn)

# ===========================================================================
# Generate all 8 test combos: {good, bad} x {single-process, fork, fork-exec, spawn}
# ===========================================================================

foreach(PROC_TYPE ${PROCESS_TYPES})
    # --- GOOD CASE: app handles signals, profiler uses --disable-signal-handlers ---
    set(WITH_HANDLING_NAME "app-with-signal-handling-${PROC_TYPE}")

    rocprofiler_add_integration_execute_test(
        ${WITH_HANDLING_NAME}
        COMMAND
            ${TIMEOUT_CMD} $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace
            --marker-trace --output-format json --disable-signal-handlers -d
            ${BASE_OUT_DIR}/good-${PROC_TYPE} -o out_%pid% --
            $<TARGET_FILE:signal-handler-test> --${PROC_TYPE} --app-signal-handler
        DEPENDS signal-handler-test
        TIMEOUT ${CTEST_TIMEOUT}
        LABELS "integration-tests;signal-handler"
        PRELOAD "${PRELOAD_ENV}"
        PASS_REGULAR_EXPRESSION "clean exit"
        FIXTURES_SETUP ${WITH_HANDLING_NAME})

    rocprofiler_add_integration_validate_test(
        ${WITH_HANDLING_NAME}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        TIMEOUT 30
        LABELS "integration-tests;signal-handler"
        FIXTURES_REQUIRED ${WITH_HANDLING_NAME}
        ARGS --output-dir ${BASE_OUT_DIR}/good-${PROC_TYPE} --mode good --process-type
             ${PROC_TYPE})

    # --- BAD CASE: app does NOT handle signals, profiler handles them ---
    set(NO_HANDLING_NAME "app-without-signal-handling-${PROC_TYPE}")

    rocprofiler_add_integration_execute_test(
        ${NO_HANDLING_NAME}
        COMMAND
            ${TIMEOUT_CMD} $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace
            --marker-trace --output-format json -d ${BASE_OUT_DIR}/bad-${PROC_TYPE} -o
            out_%pid% -- $<TARGET_FILE:signal-handler-test> --${PROC_TYPE}
            --no-app-signal-handler
        DEPENDS signal-handler-test
        TIMEOUT ${CTEST_TIMEOUT}
        LABELS "integration-tests;signal-handler"
        PRELOAD "${PRELOAD_ENV}"
        PASS_REGULAR_EXPRESSION "tool finalization"
        FIXTURES_SETUP ${NO_HANDLING_NAME})

    rocprofiler_add_integration_validate_test(
        ${NO_HANDLING_NAME}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        TIMEOUT ${CTEST_TIMEOUT}
        LABELS "integration-tests;signal-handler"
        FIXTURES_REQUIRED ${NO_HANDLING_NAME}
        ARGS --output-dir ${BASE_OUT_DIR}/bad-${PROC_TYPE} --mode bad --process-type
             ${PROC_TYPE})
endforeach()
