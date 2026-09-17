#
# rocpd command-line tests
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocpd
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(rocpd-schema-path "${CMAKE_BINARY_DIR}/share/rocprofiler-sdk-rocpd")
set(rocpd-site-packages
    "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )
set(rocpd-env "ROCPD_SCHEMA_PATH=${rocpd-schema-path}"
              "PYTHONPATH=${rocpd-site-packages}")

#########################################################################################
#
# Test the --help flag works
#
#########################################################################################

foreach(_SUBPARSER "" "-convert" "-query" "-summary")
    string(REPLACE "-" "" _CMD "${_SUBPARSER}")

    rocprofiler_add_integration_execute_test(
        rocpd${_SUBPARSER}-help
        COMMAND ${Python3_EXECUTABLE} -m rocpd ${_CMD} --help
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 120
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-env}")
endforeach()

foreach(_MODULE "csv" "pftrace" "otf2" "query" "summary")
    rocprofiler_add_integration_execute_test(
        rocpd-module-${_MODULE}-help
        COMMAND ${Python3_EXECUTABLE} -m rocpd.${_MODULE} --help
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 120
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-env}")
endforeach()

set(ROCPD_TIME_WINDOW_TEST_DB "${CMAKE_CURRENT_BINARY_DIR}/time-window.db")

rocprofiler_add_integration_execute_test(
    rocpd-time-window-db
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../rocpd-back-compat/make_db.py
        --version 3.0.4 --output ${ROCPD_TIME_WINDOW_TEST_DB}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 30
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocpd-env}"
    FIXTURES_SETUP rocpd-time-window-db)

rocprofiler_add_integration_validate_test(
    rocpd-time-window
    TEST_PATHS test_time_window.py
    CONFIG pytest.ini
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "ROCPD_SCHEMA_PATH=${rocpd-schema-path}"
    PYTHON_PATH "${rocpd-site-packages}" LIBRARY_PATH "${rocprofiler-sdk_LIB_DIR}"
    FIXTURES_REQUIRED rocpd-time-window-db
    DEPENDS rocprofiler-sdk::rocprofv3)
