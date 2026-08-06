#
# rocprofv3 HIP graph attribution integration test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-hip-graph-attribution
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(HIP_GRAPH_ATTRIBUTION_ENVIRONMENT)
if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib")
    list(APPEND HIP_GRAPH_ATTRIBUTION_ENVIRONMENT
         "LD_LIBRARY_PATH=$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib:$ENV{LD_LIBRARY_PATH}")
endif()
# Pin determinism preconditions: segmented-scheduling + direct-dispatch keep the
# per-launch node_id sequence stable across iterations.
list(APPEND HIP_GRAPH_ATTRIBUTION_ENVIRONMENT "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1"
     "AMD_DIRECT_DISPATCH=1"
     "ROCPD_SCHEMA_PATH=${CMAKE_BINARY_DIR}/share/rocprofiler-sdk-rocpd")

set(ITERATIONS 50)

# ---------------------------------------------------------------
# Primary test: kernel + graph-launch + hip-api trace
# ---------------------------------------------------------------
set(TEST_NAME "rocprofv3-test-hip-graph-attribution")
set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip-graph-attribution-out")

rocprofiler_add_integration_execute_test(
    ${TEST_NAME}
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace --kernel-trace
        --memory-copy-trace --hip-graph-trace -d ${OUTPUT_DIR} -o out --output-format csv
        -- $<TARGET_FILE:hip-graph-attribution> ${ITERATIONS}
    DEPENDS hip-graph-attribution
    TIMEOUT 120
    LABELS "integration-tests"
    ENVIRONMENT ${HIP_GRAPH_ATTRIBUTION_ENVIRONMENT}
    FIXTURES_SETUP ${TEST_NAME})

rocprofiler_add_integration_validate_test(
    ${TEST_NAME}
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --kernel-input
         ${OUTPUT_DIR}/out_kernel_trace.csv
         --memory-copy-input
         ${OUTPUT_DIR}/out_memory_copy_trace.csv
         --graph-launch-input
         ${OUTPUT_DIR}/out_graph_launch_trace.csv
         --hip-api-input
         ${OUTPUT_DIR}/out_hip_api_trace.csv
         --expected-iterations
         ${ITERATIONS}
         --expected-execs
         2
         --expected-nodes-per-launch
         6
         --expected-kernel-nodes-per-launch
         6
         --expected-memcpy-nodes-per-launch
         0
         --expected-distinct-kernels
         3
    TIMEOUT 60
    LABELS "integration-tests"
    FIXTURES_REQUIRED ${TEST_NAME})

# ---------------------------------------------------------------
# Subscription-independence variant: graph-launch only, no kernel-trace
# ---------------------------------------------------------------
set(TEST_NAME_GLO "rocprofv3-test-hip-graph-attribution-graph-launch-only")
set(OUTPUT_DIR_GLO "${CMAKE_CURRENT_BINARY_DIR}/hip-graph-attribution-glo")

rocprofiler_add_integration_execute_test(
    ${TEST_NAME_GLO}
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-graph-trace -d ${OUTPUT_DIR_GLO}
        -o out --output-format rocpd -- $<TARGET_FILE:hip-graph-attribution> ${ITERATIONS}
    DEPENDS hip-graph-attribution
    TIMEOUT 120
    LABELS "integration-tests"
    ENVIRONMENT ${HIP_GRAPH_ATTRIBUTION_ENVIRONMENT}
    FIXTURES_SETUP ${TEST_NAME_GLO})

rocprofiler_add_integration_validate_test(
    ${TEST_NAME_GLO}
    TEST_PATHS validate_graph_launch_only.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --rocpd-input ${OUTPUT_DIR_GLO}/out_results.db --expected-iterations
         ${ITERATIONS} --expected-execs 2 --expected-nodes-per-launch 6
    TIMEOUT 60
    LABELS "integration-tests"
    FIXTURES_REQUIRED ${TEST_NAME_GLO})

# ---------------------------------------------------------------
# JSON variant: same workload, JSON output to verify the new fields and HIP_GRAPH records
# flow through the JSON writer.
# ---------------------------------------------------------------
set(TEST_NAME_JSON "rocprofv3-test-hip-graph-attribution-json")
set(OUTPUT_DIR_JSON "${CMAKE_CURRENT_BINARY_DIR}/hip-graph-attribution-json")

rocprofiler_add_integration_execute_test(
    ${TEST_NAME_JSON}
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace --kernel-trace
        --hip-graph-trace -d ${OUTPUT_DIR_JSON} -o out --output-format json --
        $<TARGET_FILE:hip-graph-attribution> ${ITERATIONS}
    DEPENDS hip-graph-attribution
    TIMEOUT 120
    LABELS "integration-tests"
    ENVIRONMENT ${HIP_GRAPH_ATTRIBUTION_ENVIRONMENT}
    FIXTURES_SETUP ${TEST_NAME_JSON})

rocprofiler_add_integration_validate_test(
    ${TEST_NAME_JSON}
    TEST_PATHS validate_json.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input ${OUTPUT_DIR_JSON}/out_results.json --expected-iterations
         ${ITERATIONS} --expected-execs 2 --expected-nodes-per-launch 6
    TIMEOUT 60
    LABELS "integration-tests"
    FIXTURES_REQUIRED ${TEST_NAME_JSON})
