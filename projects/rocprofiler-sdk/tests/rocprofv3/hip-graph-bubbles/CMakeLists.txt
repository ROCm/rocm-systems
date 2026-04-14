#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-hip-graph-bubbles
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(PYTEST_ARGS)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-hip-graph-bubbles
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocprofv3-hip-graph-bubbles -o out --output-format
        rocpd --log-level env --hip-runtime-trace --kernel-trace --pmc GRBM_COUNT --
        $<TARGET_FILE:hip-graph-bubbles> 2000 10
    DEPENDS hip-graph-bubbles
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FIXTURES_SETUP hip-graph-bubbles)

rocprofiler_add_integration_validate_test(
    rocprofv3-hip-graph-bubbles
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED hip-graph-bubbles
    DISCOVERY_ARGS ${PYTEST_ARGS}
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/rocprofv3-hip-graph-bubbles/out_results.db)
