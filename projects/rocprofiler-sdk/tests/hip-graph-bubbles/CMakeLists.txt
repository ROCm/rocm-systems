#
# Standalone HIP graph bubbles integration test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-hip-graph-bubbles
    LANGUAGES CXX
    VERSION 0.0.0)

set(HIP_GRAPH_BUBBLES_STANDALONE_ENVIRONMENT)
if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib")
    list(APPEND HIP_GRAPH_BUBBLES_STANDALONE_ENVIRONMENT
         "LD_LIBRARY_PATH=$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib:$ENV{LD_LIBRARY_PATH}")
endif()

function(add_hip_graph_bubbles_standalone_test SUFFIX NUM_KERNELS NUM_ITERATIONS
         TIMEOUT_SECS)
    math(EXPR EXPECTED_DISPATCH_COUNT "${NUM_KERNELS} * ${NUM_ITERATIONS}")

    rocprofiler_add_integration_execute_test(
        hip-graph-bubbles-standalone-${SUFFIX}
        TARGET hip-graph-bubbles
        ARGS ${NUM_KERNELS} ${NUM_ITERATIONS} 256 25
        TIMEOUT ${TIMEOUT_SECS}
        LABELS "integration-tests;standalone"
        ENVIRONMENT ${HIP_GRAPH_BUBBLES_STANDALONE_ENVIRONMENT}
        PASS_REGULAR_EXPRESSION
            "Total kernel launches: ${EXPECTED_DISPATCH_COUNT};Test completed successfully"
        )
endfunction()

add_hip_graph_bubbles_standalone_test(k256-i20 256 20 30)
add_hip_graph_bubbles_standalone_test(k2000-i200 2000 200 60)
