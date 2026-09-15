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

# cmake-format: off
# add_signal_handler_test(<name> [keyword args])
#
#   <name>                            : scenario name; ctest is registered as "app-<name>-<type>"
#   PROCESS_TYPE <type>               : single-process | fork | fork-exec | spawn
#   VALIDATE_EXPECT <outcome>         : what validate.py asserts:
#                                          clean-exit: app exits cleanly via atexit, exit_marker present
#                                          flushed-markers: profiler flushed >10 markers before signal death
#   PASS_REGEX <regex>                : execute PASS_REGULAR_EXPRESSION
#   APP_EXTRA_FLAGS <flags...>        : appended to the signal-handler-test command
#                                       (e.g. --app-signal-handler / --no-app-signal-handler)
#   ROCPROFV3_EXTRA_FLAGS <flags...>  : inserted into the rocprofv3 command
#                                       (e.g. --disable-signal-handlers)
#
# Registers the {cleanup, execute, validate} triad for one scenario:
#   cleanup wipes the output dir (FIXTURES_SETUP <name>-clean)
#   before execute runs (FIXTURES_REQUIRED <name>-clean), and
#   validate runs after execute (FIXTURES_REQUIRED <name>).
# cmake-format: on
function(add_signal_handler_test NAME)
    cmake_parse_arguments(arg "" "PROCESS_TYPE;VALIDATE_EXPECT;PASS_REGEX"
                          "APP_EXTRA_FLAGS;ROCPROFV3_EXTRA_FLAGS" ${ARGN})

    set(_name "app-${NAME}-${arg_PROCESS_TYPE}")
    set(_subdir "${NAME}-${arg_PROCESS_TYPE}")
    set(_outdir ${BASE_OUT_DIR}/${_subdir})

    rocprofiler_add_integration_cleanup_test(
        ${_name}
        DIRECTORIES ${_outdir}
        LABELS "integration-tests;signal-handler"
        FIXTURES_SETUP ${_name}-clean)

    rocprofiler_add_integration_execute_test(
        ${_name}
        COMMAND
            ${TIMEOUT_CMD} $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace
            --marker-trace --output-format json ${arg_ROCPROFV3_EXTRA_FLAGS} -d ${_outdir}
            -o out_%pid% -- $<TARGET_FILE:signal-handler-test> --${arg_PROCESS_TYPE}
            ${arg_APP_EXTRA_FLAGS}
        DEPENDS signal-handler-test
        TIMEOUT ${CTEST_TIMEOUT}
        LABELS "integration-tests;signal-handler"
        PRELOAD "${PRELOAD_ENV}"
        PASS_REGULAR_EXPRESSION "${arg_PASS_REGEX}"
        FIXTURES_REQUIRED ${_name}-clean
        FIXTURES_SETUP ${_name})

    rocprofiler_add_integration_validate_test(
        ${_name}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        TIMEOUT ${CTEST_TIMEOUT}
        LABELS "integration-tests;signal-handler"
        FIXTURES_REQUIRED ${_name}
        ARGS --output-dir ${_outdir} --expect ${arg_VALIDATE_EXPECT} --process-type
             ${arg_PROCESS_TYPE})
endfunction()

# good case: app installs its own signal handler, rocprofv3 runs with
# --disable-signal-handlers, so the app exits cleanly through atexit.
add_signal_handler_test(
    with-signal-handling
    PROCESS_TYPE single-process
    VALIDATE_EXPECT clean-exit
    PASS_REGEX "clean exit"
    APP_EXTRA_FLAGS --app-signal-handler
    ROCPROFV3_EXTRA_FLAGS --disable-signal-handlers)

add_signal_handler_test(
    with-signal-handling
    PROCESS_TYPE fork
    VALIDATE_EXPECT clean-exit
    PASS_REGEX "clean exit"
    APP_EXTRA_FLAGS --app-signal-handler
    ROCPROFV3_EXTRA_FLAGS --disable-signal-handlers)

add_signal_handler_test(
    with-signal-handling
    PROCESS_TYPE fork-exec
    VALIDATE_EXPECT clean-exit
    PASS_REGEX "clean exit"
    APP_EXTRA_FLAGS --app-signal-handler
    ROCPROFV3_EXTRA_FLAGS --disable-signal-handlers)

add_signal_handler_test(
    with-signal-handling
    PROCESS_TYPE spawn
    VALIDATE_EXPECT clean-exit
    PASS_REGEX "clean exit"
    APP_EXTRA_FLAGS --app-signal-handler
    ROCPROFV3_EXTRA_FLAGS --disable-signal-handlers)

# bad case: app does NOT handle signals, so rocprofv3's handler catches the signal and
# flushes profiling data before the process dies.
add_signal_handler_test(
    without-signal-handling
    PROCESS_TYPE single-process
    VALIDATE_EXPECT flushed-markers
    PASS_REGEX "tool finalization"
    APP_EXTRA_FLAGS --no-app-signal-handler)

add_signal_handler_test(
    without-signal-handling
    PROCESS_TYPE fork
    VALIDATE_EXPECT flushed-markers
    PASS_REGEX "tool finalization"
    APP_EXTRA_FLAGS --no-app-signal-handler)

add_signal_handler_test(
    without-signal-handling
    PROCESS_TYPE fork-exec
    VALIDATE_EXPECT flushed-markers
    PASS_REGEX "tool finalization"
    APP_EXTRA_FLAGS --no-app-signal-handler)

add_signal_handler_test(
    without-signal-handling
    PROCESS_TYPE spawn
    VALIDATE_EXPECT flushed-markers
    PASS_REGEX "tool finalization"
    APP_EXTRA_FLAGS --no-app-signal-handler)

# Coordinated-shutdown case: profiler signal handlers are ON while the app runs its own
# coordinated multi-process shutdown (the sglang/vLLM tensor-parallel pattern). Worker
# children ignore SIGINT and only exit once the parent tells them to. The profiler must
# re-raise to the app's handler before waiting for children to exit, else it deadlocks
# because parent waits for children that are waiting for the parent and the test hangs,
# which triggers a ctest TIMEOUT.
#
# NOTE: unlike the good case, this does NOT pass --disable-signal-handlers, so the
# profiler's handlers are exercised alongside the app's own --app-signal-handler.
add_signal_handler_test(
    coordinated-shutdown
    PROCESS_TYPE fork
    VALIDATE_EXPECT flushed-markers
    PASS_REGEX "clean exit"
    APP_EXTRA_FLAGS --app-signal-handler)
