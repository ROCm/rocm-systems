#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-list-metrics
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_configure_pytest_files(
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    CONFIG pytest.ini
    COPY input.json)

# WSL2/DXG has no /dev/kfd and schedules GPU work through the dxg path. Under WSL2/DXG
# these workloads run slower in practice and need more timeout headroom (root cause TBD),
# so scale only there for the traced list-avail workload.
rocprofiler_sdk_wsl_timeout_scale(_LIST_AVAIL_TRACE_TIMEOUT BASE 60)

# basic-metrics
rocprofiler_add_integration_execute_test(
    rocprofv3-test-list-avail
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
            ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2 -o metrics --list-avail
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 60
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPD_LOG_LEVEL=info")

# list-metrics-stdout
rocprofiler_add_integration_execute_test(
    rocprofv3-test-list-avail-std-out
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
            ${CMAKE_CURRENT_BINARY_DIR}/input.json
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPD_LOG_LEVEL=info"
    PASS_REGULAR_EXPRESSION
        "GPU:[0-9]*\\n*;Name:\\t[a-zA-Z_]*\\n;counter_name:\\t[a-zA-Z_]*\\n;description:\\t(.*)\\n*;expression:\\t(.)*\\n*;block:\\t[a-zA-Z]*\\n*;dimension_name:\\t([A-Z_]*)\\n*; dimension_instances:\\t([[0-9]*:[0-9]*\\])*\\n*;method:(.*)\\n*;unit:(.*)\\n*;min_interval:[0-9]*\\n*;max_interval:[0-9]*\\n*;flags:(.*)\\n*"
    )

# list-metrics with tracing
rocprofiler_add_integration_execute_test(
    rocprofv3-test-list-avail-trace
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2 -o metrics --list-avail
        --sys-trace --log-level warning -- $<TARGET_FILE:transpose>
    DEPENDS transpose
    TIMEOUT ${_LIST_AVAIL_TRACE_TIMEOUT}
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPD_LOG_LEVEL=info")
