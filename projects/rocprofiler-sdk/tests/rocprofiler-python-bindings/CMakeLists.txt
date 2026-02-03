#
# Python unit tests for rocprofiler Python bindings
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofiler-python-bindings
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(NOT Python3_EXECUTABLE)
    find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)
endif()

# Copy test files to build directory
set(ROCPROFILER_PYTHON_TEST_FILES conftest.py pytest.ini test_counters.py
                                  test_profiler.py test_records.py test_integration.py)

foreach(_TEST_FILE ${ROCPROFILER_PYTHON_TEST_FILES})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_TEST_FILE}
                   ${CMAKE_CURRENT_BINARY_DIR}/${_TEST_FILE} COPYONLY)
endforeach()

# Unit tests (no GPU required)
rocprofiler_add_integration_validate_test(
    rocprofiler-python-unit-tests
    TEST_PATHS test_counters.py test_profiler.py test_records.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "unittests;python-bindings"
    TIMEOUT 60
    ENVIRONMENT
        "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FAIL_REGULAR_EXPRESSION "AssertionError")

# Integration tests (GPU required)
rocprofiler_add_integration_validate_test(
    rocprofiler-python-integration-tests
    TEST_PATHS test_integration.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests;python-bindings;gpu"
    TIMEOUT 120
    ENVIRONMENT
        "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    ARGS --run-gpu
    FAIL_REGULAR_EXPRESSION "AssertionError"
    UNSTABLE)
