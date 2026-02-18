#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-trace
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -M --hsa-trace --kernel-trace
        --memory-copy-trace --marker-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o
        cmdl-input/out --output-format pftrace csv json rocpd --log-level ${LOG_LEVEL} --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-trace
    FAIL_REGULAR_EXPRESSION
        "HSA_API|HIP_API|HIP_COMPILER_API|MARKER_CORE_API|MARKER_CONTROL_API|MARKER_NAME_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

rocprofiler_add_integration_validate_test(
    rocprofv3-test-trace
    TEST_PATHS validate.py
    COPY conftest.py input_systrace.json input_trace.json
    CONFIG pytest.ini
    ARGS --hsa-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_hsa_api_trace.csv
         --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_kernel_trace.csv
         --memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_memory_copy_trace.csv
         --marker-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_marker_api_trace.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_agent_info.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_results.pftrace
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/cmdl-input/out_results.db
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    FIXTURES_REQUIRED rocprofv3-test-trace)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-trace-input-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_trace.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace --log-level ${LOG_LEVEL} --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-trace-input-json
    FAIL_REGULAR_EXPRESSION
        "HSA_API|HIP_API|HIP_COMPILER_API|MARKER_CORE_API|MARKER_CONTROL_API|MARKER_NAME_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

rocprofiler_add_integration_validate_test(
    rocprofv3-test-trace-input-json
    TEST_PATHS validate.py
    COPY conftest.py input_systrace.json input_trace.json
    CONFIG pytest.ini
    ARGS --hsa-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_hsa_api_trace.csv
         --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_kernel_trace.csv
         --memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_memory_copy_trace.csv
         --marker-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_marker_api_trace.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_agent_info.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_results.pftrace
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/json-input/out_results.db
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-trace-input-json)

# sys-trace test: tests --sys-trace command with mangled kernel names and validates
# generated files

rocprofiler_add_integration_execute_test(
    rocprofv3-test-systrace
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-systrace -o cmdl-input/out --output-format
        pftrace csv json rocpd -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-systrace
    FAIL_REGULAR_EXPRESSION
        "HSA_API|HIP_API|HIP_COMPILER_API|MARKER_CORE_API|MARKER_CONTROL_API|MARKER_NAME_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

rocprofiler_add_integration_validate_test(
    rocprofv3-test-systrace
    TEST_PATHS validate.py
    COPY conftest.py input_systrace.json input_trace.json
    CONFIG pytest.ini
    DISCOVERY_ARGS -k "not test_hsa_api_trace"
    ARGS --hsa-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_hsa_api_trace.csv
         --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_kernel_trace.csv
         --memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_memory_copy_trace.csv
         --marker-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_marker_api_trace.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_agent_info.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_results.pftrace
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/cmdl-input/out_results.db
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-systrace)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-systrace-input-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_systrace.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-systrace -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-systrace-input-json
    FAIL_REGULAR_EXPRESSION
        "HSA_API|HIP_API|HIP_COMPILER_API|MARKER_CORE_API|MARKER_CONTROL_API|MARKER_NAME_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

rocprofiler_add_integration_validate_test(
    rocprofv3-test-systrace-input-json
    TEST_PATHS validate.py
    COPY conftest.py input_systrace.json input_trace.json
    CONFIG pytest.ini
    DISCOVERY_ARGS -k "not test_hsa_api_trace"
    ARGS --hsa-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_hsa_api_trace.csv
         --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_kernel_trace.csv
         --memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_memory_copy_trace.csv
         --marker-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_marker_api_trace.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_agent_info.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_results.pftrace
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/json-input/out_results.db
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-systrace-input-json)
