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
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-runtime-trace
        --hip-compiler-trace --hsa-core-trace --hsa-amd-trace --hsa-image-trace
        --hsa-finalizer-trace --kernel-trace --memory-copy-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out $<TARGET_FILE:hip-in-libraries>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-trace-hip-in-libraries-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${tracing-env}"
        FAIL_REGULAR_EXPRESSION
        "HSA_CORE_API|HSA_AMD_EXT_API|HSA_IMAGE_EXT_API|HSA_FINALIZER_EXT_API|HIP_API|HIP_COMPILER_API|KERNEL_DISPATCH|CODE_OBJECT"
    )

foreach(FILENAME validate.py conftest.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

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
        ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_memory_copy_trace.csv)

set(VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_memory_copy_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hsa_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_hip_api_trace.csv
    ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_kernel_trace.csv)

set_tests_properties(
    rocprofv3-test-trace-hip-in-libraries-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-trace-hip-in-libraries-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               ATTACHED_FILES_ON_FAIL
               "${VALIDATION_FILES}")
