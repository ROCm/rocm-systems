#
# rocprofv3 rocpd tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocpd-scratch
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(rocprofv3-rocpd-scratch-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    "OMPI_ALLOW_RUN_AS_ROOT=1"
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py)

find_package(Python3 REQUIRED)

#########################################################################################
#
# generate rocpd database and json for comparison
#
#########################################################################################

add_test(
    NAME rocprofv3-test-rocpd-scratch-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --scratch-memory-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data -o out --output-format rocpd json
        --runtime-trace --kernel-rename --output-config -- $<TARGET_FILE:scratch-memory>)

set_tests_properties(
    rocprofv3-test-rocpd-scratch-execute
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;rocpd"
               ENVIRONMENT
               "${rocprofv3-rocpd-scratch-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-rocpd-scratch)

#########################################################################################
#
# perfetto generate
#
#########################################################################################

add_test(
    NAME rocprofv3-test-rocpd-scratch-perfetto-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db)

set_tests_properties(
    rocprofv3-test-rocpd-scratch-perfetto-generation
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;rocpd"
               ENVIRONMENT
               "${rocprofv3-rocpd-scratch-env}"
               DEPENDS
               "rocprofv3-test-rocpd-scratch-execute"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-rocpd-scratch-generation
               FIXTURES_REQUIRED
               rocprofv3-test-rocpd-scratch)

#########################################################################################
#
# Validation
#
#########################################################################################

set(VALIDATION_DEPENDS
    rocprofv3-test-rocpd-scratch-perfetto-generation)

add_test(
    NAME rocprofv3-test-rocpd-scratch-validation
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.json --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_results.pftrace)

set_tests_properties(
    rocprofv3-test-rocpd-scratch-validation
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;rocpd"
               ENVIRONMENT
               "${rocprofv3-rocpd-scratch-env}"
               DEPENDS 
               rocprofv3-test-rocpd-scratch-perfetto-generation
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_REQUIRED
               rocprofv3-test-rocpd-scratch-generation)
