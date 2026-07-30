#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-thread-trace
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Single dispatch test
rocprofiler_add_integration_execute_test(
    thread-trace-api-single-test
    TARGET thread-trace-test-app
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:thread-trace-single-tool>")

# Multi dispatch test
rocprofiler_add_integration_execute_test(
    thread-trace-api-multi-test
    TARGET thread-trace-test-app
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:thread-trace-multi-tool>")

# Agent profiling test
rocprofiler_add_integration_execute_test(
    thread-trace-api-agent-test
    TARGET thread-trace-test-app
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:thread-trace-agent-tool>")

# Start before hsa_init(): verifies the request is cached and replayed once HSA registers.
rocprofiler_add_integration_execute_test(
    thread-trace-api-agent-before-hsa-init-test
    TARGET thread-trace-test-app
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:thread-trace-agent-tool>"
    ENVIRONMENT "ATT_START_BEFORE_HSA_INIT=1")

# Start+stop before hsa_init(): verifies the cached request is dropped and not replayed.
rocprofiler_add_integration_execute_test(
    thread-trace-api-agent-start-stop-before-hsa-init-test
    TARGET thread-trace-test-app
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:thread-trace-agent-tool>"
    ENVIRONMENT "ATT_START_BEFORE_HSA_INIT=1;ATT_STOP_BEFORE_HSA_INIT=1")

# Feature introduced in ROCm 7.1, however some pre-release CI builds for 7.1 dont have
# this feature in aqlprofile. Disable the next tests for ROCm < 7.2.
set(IS_DISABLED True)
find_package(hip CONFIG REQUIRED)
if(hip_FOUND)
    if(${hip_VERSION_MAJOR} EQUAL 7 AND ${hip_VERSION_MINOR} GREATER 1)
        set(IS_DISABLED False)
    elseif(${hip_VERSION_MAJOR} GREATER 7)
        set(IS_DISABLED False)
    endif()
endif()

# Test large buffer sizes. 5120 == 5GB
rocprofiler_add_integration_execute_test(
    thread-trace-api-large-buffer-test
    TARGET thread-trace-test-app
    TIMEOUT 30
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:thread-trace-agent-tool>"
    ENVIRONMENT "ATT_LARGE_BUFFER_TEST=1"
    DISABLED ${IS_DISABLED})

# Test occupancy mode
rocprofiler_add_integration_execute_test(
    thread-trace-api-extra-args
    TARGET thread-trace-test-app
    TIMEOUT 10
    PRELOAD "$<TARGET_FILE:thread-trace-agent-tool>"
    ENVIRONMENT "ATT_NODETAIL=1"
    DISABLED ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})

rocprofiler_sdk_sqtt_triple_buffer_disabled(TRIPLE_BUFFER_DISABLED)
set(TRIPLE_BUFFER_DISABLED ${TRIPLE_BUFFER_DISABLED} OR ${IS_DISABLED})

# Checks for data integrity with ttracedata
rocprofiler_add_integration_execute_test(
    thread-trace-api-triple-buffer-consistency-test
    TARGET thread-trace-test-app
    TIMEOUT 120
    PRELOAD "$<TARGET_FILE:thread-trace-triple-buffer-tool>"
    ENVIRONMENT "TRIPLEBUFFER=1;ATT_NODETAIL=1;ATT_SLOW_CALLBACK=0"
    DISABLED ${TRIPLE_BUFFER_DISABLED})

# Collects a lot of data, and check if parsing works successfully
rocprofiler_add_integration_execute_test(
    thread-trace-api-triple-buffer-hammer-test
    TARGET thread-trace-test-app
    TIMEOUT 240
    PRELOAD "$<TARGET_FILE:thread-trace-triple-buffer-tool>"
    ENVIRONMENT "TRIPLEBUFFER=1;ATT_NODETAIL=0;ATT_SLOW_CALLBACK=0"
    DISABLED ${TRIPLE_BUFFER_DISABLED})

# Tests for proper handling of when the tool is very slow
rocprofiler_add_integration_execute_test(
    thread-trace-api-triple-buffer-slow-test
    TARGET thread-trace-test-app
    TIMEOUT 120
    PRELOAD "$<TARGET_FILE:thread-trace-triple-buffer-tool>"
    ENVIRONMENT "TRIPLEBUFFER=1;ATT_NODETAIL=1;ATT_SLOW_CALLBACK=1"
    DISABLED ${TRIPLE_BUFFER_DISABLED})

# Some runtime versions have a problem with this. Disabling until that is fully resolved.
# Tests for thread trace not paused during application run (global destructor)
rocprofiler_add_integration_execute_test(
    thread-trace-api-triple-buffer-no-pause-test
    TARGET thread-trace-test-app
    TIMEOUT 120
    PRELOAD "$<TARGET_FILE:thread-trace-triple-buffer-tool>"
    ENVIRONMENT "TRIPLEBUFFER=1;EXECUTE_PAUSE=0"
    DISABLED ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS} OR ${TRIPLE_BUFFER_DISABLED})

# Tests for thread trace starting and stopping multiple times during the application
rocprofiler_add_integration_execute_test(
    thread-trace-api-triple-buffer-multiple-cmds-test
    TARGET thread-trace-test-app
    TIMEOUT 120
    PRELOAD "$<TARGET_FILE:thread-trace-triple-buffer-tool>"
    ENVIRONMENT "TRIPLEBUFFER=1;STARTSTOP=1;ATT_NODETAIL=1"
    DISABLED ${TRIPLE_BUFFER_DISABLED})
