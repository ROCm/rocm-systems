#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-tracing-hip-in-libraries
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

add_test(
    NAME rocprofv3-test-trace-hip-in-libraries-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-runtime-trace --hsa-core-trace
        --hsa-amd-trace --marker-trace --kernel-trace --memory-copy-trace --stats
        --output-format csv -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out --
        $<TARGET_FILE:hip-in-libraries>)

add_test(
    NAME rocprofv3-test-trace-hip-in-libraries-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-runtime-trace --hsa-core-trace
        --hsa-amd-trace --marker-trace --kernel-trace --memory-copy-trace --stats
        --output-format JSON -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out --
        $<TARGET_FILE:hip-in-libraries>)

add_test(
    NAME rocprofv3-test-trace-hip-in-libraries-pftrace-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-runtime-trace --hsa-core-trace
        --hsa-amd-trace --marker-trace --kernel-trace --memory-copy-trace --stats
        --output-format pftrace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out --
        $<TARGET_FILE:hip-in-libraries>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-trace-hip-in-libraries-execute
    PROPERTIES
        TIMEOUT
        100
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${tracing-env}"
        FAIL_REGULAR_EXPRESSION
        "HSA_CORE_API|HSA_AMD_EXT_API|HSA_IMAGE_EXT_API|HSA_FINALIZER_EXT_API|HIP_API|HIP_COMPILER_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

set_tests_properties(
    rocprofv3-test-trace-hip-in-libraries-json-execute
    PROPERTIES
        TIMEOUT
        100
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${tracing-env}"
        FAIL_REGULAR_EXPRESSION
        "HSA_CORE_API|HSA_AMD_EXT_API|HSA_IMAGE_EXT_API|HSA_FINALIZER_EXT_API|HIP_API|HIP_COMPILER_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

set_tests_properties(
    rocprofv3-test-trace-hip-in-libraries-pftrace-execute
    PROPERTIES
        TIMEOUT
        100
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${tracing-env}"
        FAIL_REGULAR_EXPRESSION
        "HSA_CORE_API|HSA_AMD_EXT_API|HSA_IMAGE_EXT_API|HSA_FINALIZER_EXT_API|HIP_API|HIP_COMPILER_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

add_test(
    NAME rocprofv3-test-trace-hip-in-libraries-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --hsa-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hsa_api_trace.csv
        --hip-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hip_api_trace.csv
        --kernel-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_kernel_trace.csv
        --memory-copy-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_memory_copy_trace.csv
        --marker-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_marker_api_trace.csv
        --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_agent_info.csv
        --kernel-stats
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_kernel_stats.csv
        --hip-stats ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hip_stats.csv
        --hsa-stats ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hsa_stats.csv
        --memory-copy-stats
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_memory_copy_stats.csv
        --json-input ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_results.pftrace)

set(VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_results.pftrace
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_results.json
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_memory_copy_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hsa_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hip_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_kernel_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_agent_info.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_kernel_stats.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hip_stats.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hsa_stats.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_memory_copy_stats.csv)

set_tests_properties(
    rocprofv3-test-trace-hip-in-libraries-validate
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        DEPENDS
        "rocprofv3-test-trace-hip-in-libraries-execute;rocprofv3-test-trace-hip-in-libraries-json-execute;rocprofv3-test-trace-hip-in-libraries-pftrace-execute"
        FAIL_REGULAR_EXPRESSION
        "AssertionError"
        ATTACHED_FILES_ON_FAIL
        "${VALIDATION_FILES}")
