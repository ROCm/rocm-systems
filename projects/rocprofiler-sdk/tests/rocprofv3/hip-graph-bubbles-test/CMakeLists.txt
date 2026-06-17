#
# rocprofv3 HIP graph bubble regression tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-hip-graph-bubbles
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(HIP_GRAPH_BUBBLES_ENVIRONMENT)
if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib")
    list(APPEND HIP_GRAPH_BUBBLES_ENVIRONMENT
         "LD_LIBRARY_PATH=$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib:$ENV{LD_LIBRARY_PATH}")
endif()

function(add_hip_graph_bubbles_kernel_trace_test SUFFIX NUM_KERNELS NUM_ITERATIONS
         TIMEOUT_SECS)
    set(TEST_NAME "rocprofv3-test-hip-graph-bubbles-${SUFFIX}")
    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip-graph-bubbles-${SUFFIX}")
    math(EXPR EXPECTED_DISPATCH_COUNT "${NUM_KERNELS} * ${NUM_ITERATIONS}")

    rocprofiler_add_integration_execute_test(
        ${TEST_NAME}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace --kernel-trace -d
            ${OUTPUT_DIR} -o out --output-format csv pftrace --
            $<TARGET_FILE:hip-graph-bubbles> ${NUM_KERNELS} ${NUM_ITERATIONS}
            ${NUM_ITERATIONS} 256 25
        DEPENDS hip-graph-bubbles
        TIMEOUT ${TIMEOUT_SECS}
        LABELS "integration-tests"
        ENVIRONMENT ${HIP_GRAPH_BUBBLES_ENVIRONMENT}
        FIXTURES_SETUP ${TEST_NAME}
        FAIL_REGULAR_EXPRESSION
            "HSA_API|HIP_API|HIP_COMPILER_API|MARKER_CORE_API|MARKER_CONTROL_API|MARKER_NAME_API|KERNEL_DISPATCH|CODE_OBJECT"
        )

    rocprofiler_add_integration_validate_test(
        ${TEST_NAME}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        ARGS --kernel-input
             ${CMAKE_CURRENT_BINARY_DIR}/hip-graph-bubbles-${SUFFIX}/out_kernel_trace.csv
             --expected-dispatch-count
             ${EXPECTED_DISPATCH_COUNT}
             --expected-kernels
             ${NUM_KERNELS}
             --expected-iterations
             ${NUM_ITERATIONS}
        TIMEOUT ${TIMEOUT_SECS}
        LABELS "integration-tests"
        FIXTURES_REQUIRED ${TEST_NAME})
endfunction()

add_hip_graph_bubbles_kernel_trace_test(k256-i20 256 20 60)
add_hip_graph_bubbles_kernel_trace_test(k256-i200 256 200 120)
add_hip_graph_bubbles_kernel_trace_test(k2000-i200 2000 200 180)
