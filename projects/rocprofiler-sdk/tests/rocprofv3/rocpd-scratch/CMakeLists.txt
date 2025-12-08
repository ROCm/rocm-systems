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

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-scratch
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --scratch-memory-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data -o out --output-format rocpd json
        --runtime-trace --kernel-rename -- $<TARGET_FILE:scratch-memory>
    DEPENDS scratch-memory
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-scratch-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-scratch)

#########################################################################################
#
# perfetto generate
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-scratch-perfetto-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-scratch-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-scratch-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd-scratch)

#########################################################################################
#
# Validation
#
#########################################################################################

rocprofiler_add_integration_validate_test(
    rocprofv3-test-rocpd-scratch
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.json
         --pftrace-input ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_results.pftrace
    TIMEOUT 120
    LABELS "integration-tests;rocpd" PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-scratch-env}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-scratch-generation)
