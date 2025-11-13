#
# rocprofv3 tool tests for kernel duration ns
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kernel-duration-ns
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(rocprofv3-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py)

add_test(
    NAME rocprofv3-test-kernel-duration-ns-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-duration-ns -o out --output-format csv
        --kernel-trace -- $<TARGET_FILE:simple-transpose>)

set_tests_properties(
    rocprofv3-test-kernel-duration-ns-execute
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-duration-ns"
               ENVIRONMENT
               "${rocprofv3-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-duration-ns-run)

add_test(
    NAME rocprofv3-test-kernel-duration-ns-validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --csv-input
            ${CMAKE_CURRENT_BINARY_DIR}/kernel-duration-ns/out_kernel_trace.csv)

set_tests_properties(
    rocprofv3-test-kernel-duration-ns-validation
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-duration-ns"
               ENVIRONMENT
               "${rocprofv3-env}"
               DEPENDS
               "rocprofv3-test-kernel-duration-ns-execute"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-duration-ns-run)
