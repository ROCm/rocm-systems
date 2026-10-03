#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-kernel-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# signal-less completion is pinned off so that no record may report a pipe
set(kernel-tracing-env
    "ROCPROFILER_TOOL_DISABLE_PERFETTO=1"
    "ROCPROFILER_TOOL_OUTPUT_FILE=kernel-tracing-test.json"
    "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS=0")

rocprofiler_add_integration_execute_test(
    kernel-tracing
    TARGET reproducible-runtime
    TIMEOUT 100
    LABELS "integration-tests"
    ENVIRONMENT "${kernel-tracing-env}"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    FIXTURES_SETUP kernel-tracing)

# Stable structural/logical kernel-tracing validation tests. test_total_runtime is
# excluded here and registered as its own (unstable) test below.
rocprofiler_add_integration_validate_test(
    kernel-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED kernel-tracing
    DISCOVERY_ARGS -k "not test_total_runtime"
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test.json)

rocprofiler_add_integration_validate_test(
    kernel-tracing-total-runtime
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED kernel-tracing
    DISCOVERY_ARGS -k "test_total_runtime"
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test.json
    DISABLED_CODECOV
    DISABLED_MEMCHECKS "(Address|Thread|UndefinedBehavior)Sanitizer"
    UNSTABLE)

# Signal-less completion is requested, but whether it engages depends on the KFD and the
# firmware, which cannot be probed from here. The tool's signal-less summary line on
# stderr reports how many dispatches it completed, so stderr is kept in a log for the
# validator. The json-tool is preloaded only into the application, not into the wrapping
# shell.
set(kernel-tracing-signal-less-log
    ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-signal-less.log)

set(kernel-tracing-signal-less-env
    "ROCPROFILER_TOOL_DISABLE_PERFETTO=1"
    "ROCPROFILER_TOOL_OUTPUT_FILE=kernel-tracing-signal-less-test.json"
    "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS=1" "ROCPROFILER_LOG_LEVEL=warning")

rocprofiler_add_integration_execute_test(
    kernel-tracing-signal-less
    COMMAND
        sh -c
        "LD_PRELOAD=\"\${LD_PRELOAD:+\${LD_PRELOAD}:}$<TARGET_FILE:rocprofiler-sdk-json-tool>\" $<TARGET_FILE:reproducible-runtime> 2>${kernel-tracing-signal-less-log}; rc=\$?; cat ${kernel-tracing-signal-less-log} >&2; exit \$rc"
    DEPENDS reproducible-runtime
    TIMEOUT 100
    LABELS "integration-tests"
    ENVIRONMENT "${kernel-tracing-signal-less-env}"
    FIXTURES_SETUP kernel-tracing-signal-less)

# Only the pipe-id test: where signal-less engages but the firmware never reports
# completions, dispatches go unrecorded and the structural tests above would fail.
rocprofiler_add_integration_validate_test(
    kernel-tracing-signal-less
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED kernel-tracing-signal-less
    DISCOVERY_ARGS -k "test_kernel_dispatch_pipe_ids"
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-signal-less-test.json
         --signal-less-log ${kernel-tracing-signal-less-log})
