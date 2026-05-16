#
# rocprofv3 rocpd sample counter track tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocpd-sample-tracks
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(rocprofv3-rocpd-sample-tracks-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    "OMPI_ALLOW_RUN_AS_ROOT=1"
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py
                                                          create_sample_db_schema3_0.py)

find_package(Python3 REQUIRED)

#########################################################################################
#
# generate synthetic rocpd database (schema 3.0, no GPU hardware required)
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-sample-tracks-3-0-create-db
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/create_sample_db_schema3_0.py
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/rocpd-sample-tracks-data
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 60
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-sample-tracks-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-sample-tracks)

#########################################################################################
#
# convert the synthetic database to a pftrace file
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-sample-tracks-3-0-pftrace-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-sample-tracks-output -o sample_tracks_schema3_0
        -i ${CMAKE_CURRENT_BINARY_DIR}/rocpd-sample-tracks-data/sample_tracks_schema3_0.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 60
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-sample-tracks-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-sample-tracks-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd-sample-tracks)

#########################################################################################
#
# Validation
#
#########################################################################################

rocprofiler_add_integration_validate_test(
    rocprofv3-test-rocpd-sample-tracks-3-0
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-sample-tracks-output/sample_tracks_schema3_0_results.pftrace
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-sample-tracks-env}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-sample-tracks-generation)
