#
# rocprofv3 python bindings for roctx test(s)
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-python-bindings
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(NOT Python3_EXECUTABLE)
    find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    roctx-python-bindings
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --marker-trace --summary -u sec -d
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings -o out --output-format csv json
        pftrace --log-level config -- ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/marker.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests;python-bindings"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT
        "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_SETUP roctx-python-bindings)

rocprofiler_add_integration_validate_test(
    roctx-python-bindings
    TEST_PATHS validate.py
    COPY conftest.py marker.py
    CONFIG pytest.ini
    LABELS "integration-tests;python-bindings"
    TIMEOUT 45
    FIXTURES_REQUIRED roctx-python-bindings
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_agent_info.csv
         --marker-input
         ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_marker_api_trace.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_results.pftrace
    FAIL_REGULAR_EXPRESSION "AssertionError")
