#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

add_test(
    NAME rocprofv3-test-trace-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -M --hsa-trace --kernel-trace
        --memory-copy-trace --marker-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o
        out --output-format pftrace csv json --log-level ${LOG_LEVEL} --
        $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-trace-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${tracing-env}"
        FAIL_REGULAR_EXPRESSION
        "HSA_API|HIP_API|HIP_COMPILER_API|MARKER_CORE_API|MARKER_CONTROL_API|MARKER_NAME_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

add_test(
    NAME rocprofv3-test-trace-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --hsa-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_hsa_api_trace.csv
        --kernel-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_kernel_trace.csv
        --memory-copy-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_memory_copy_trace.csv
        --marker-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_marker_api_trace.csv
        --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_agent_info.csv
        --json-input ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_results.pftrace)

set(VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_results.pftrace
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_results.json
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_memory_copy_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_hsa_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_kernel_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_marker_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-trace-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-trace-execute"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               ATTACHED_FILES_ON_FAIL
               "${VALIDATION_FILES}")

# sys-trace test: tests --sys-trace command with mangled kernel names and validates
# generated files

add_test(
    NAME rocprofv3-test-systrace-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-systrace -o out --output-format pftrace csv
        json -- $<TARGET_FILE:simple-transpose>)

set_tests_properties(
    rocprofv3-test-systrace-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${tracing-env}"
        FAIL_REGULAR_EXPRESSION
        "HSA_API|HIP_API|HIP_COMPILER_API|MARKER_CORE_API|MARKER_CONTROL_API|MARKER_NAME_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

add_test(
    NAME rocprofv3-test-systrace-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        "not test_hsa_api_trace" --hsa-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_hsa_api_trace.csv
        --kernel-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_kernel_trace.csv
        --memory-copy-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_memory_copy_trace.csv
        --marker-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_marker_api_trace.csv
        --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_agent_info.csv
        --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_results.pftrace)

set(SYS_VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_results.pftrace
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_results.json
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_memory_copy_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_hsa_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_kernel_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_marker_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-systrace/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-systrace-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-systrace-execute"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               ATTACHED_FILES_ON_FAIL
               "${SYS_VALIDATION_FILES}")
