#
# rocprofv3 python bindings for roctx test(s)
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-python-binding
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(NOT Python3_EXECUTABLE)
    find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
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
    NAME rocprofv3-trace-roctx-python-bindings-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --marker-trace --summary -u sec -d
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings -o out --output-format csv
        json pftrace --log-level ${LOG_LEVEL} -- ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/marker.py)

set_tests_properties(
    rocprofv3-trace-roctx-python-bindings-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${tracing-env}")

add_test(
    NAME rocprofv3-trace-roctx-python-bindings-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_agent_info.csv
        --marker-input
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_marker_api_trace.csv
        --json-input ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/marker-python-bindings/out_results.pftrace)

set_tests_properties(
    rocprofv3-trace-roctx-python-bindings-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-trace-roctx-python-bindings-execute"
               DISABLED
               "${TRANSPOSE_ROCTRACER_ROCTX_DISABLED}"
               FAIL_REGULAR_EXPRESSION
               "AssertionError")
