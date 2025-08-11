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

set(tracing-env
    "${PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

rocprofiler_configure_pytest_files(CONFIG pytest.ini marker.py COPY validate.py
                                                                    conftest.py)

add_test(
    NAME test-roctx-python-bindings-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --marker-trace --summary -u sec -d
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings -o out --output-format csv
        json pftrace --log-level config -- ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/marker.py)

set_tests_properties(
    test-roctx-python-bindings-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests;python-bindings" ENVIRONMENT
               "${tracing-env}")

add_test(
    NAME test-roctx-python-bindings-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_agent_info.csv
        --marker-input
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_marker_api_trace.csv
        --json-input ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_results.pftrace)

set_tests_properties(
    test-roctx-python-bindings-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests;python-bindings" DEPENDS
               "test-roctx-python-bindings-execute" FAIL_REGULAR_EXPRESSION
               "AssertionError")
