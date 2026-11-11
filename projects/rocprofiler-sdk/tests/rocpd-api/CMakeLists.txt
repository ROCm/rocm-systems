#
# rocpd API tests: C++ rocpd-api-test app (main) plus optional pytest schema checks
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocpd-api
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(rocpd-api-schema-path "${CMAKE_BINARY_DIR}/share/rocprofiler-sdk-rocpd")
set(rocpd-api-site-packages
    "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )
set(rocpd-api-env
    "ROCPD_SCHEMA_PATH=${rocpd-api-schema-path}"
    "PYTHONPATH=${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages:${rocpd-api-site-packages}"
    )

################################################################################
# Main test: run C++ rocpd-api-test (rocpd_get_version + rocpd_sql_load_schema)
################################################################################
rocprofiler_add_integration_execute_test(
    rocpd-api-cpp-interface-test
    COMMAND $<TARGET_FILE:rocpd-api-test> --all
    DEPENDS rocpd-api-test
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocpd-api-env}")

################################################################################
# Regression test: the library has to find the schema directory that ships with it,
# not the one belonging to whichever librocprofiler-sdk-rocpd.so happens to be first
# in the global symbol scope. Interpose the exported entry point from a prefix that
# has no share/rocprofiler-sdk-rocpd and give the test no ROCPD_SCHEMA_PATH to fall
# back on, so the only way the schema loads is if the library located itself.
################################################################################
find_package(rocprofiler-sdk-rocpd REQUIRED)

add_library(rocpd-api-stale-rocpd-decoy SHARED)
target_sources(rocpd-api-stale-rocpd-decoy PRIVATE stale-rocpd-decoy.cpp)
target_compile_options(rocpd-api-stale-rocpd-decoy PRIVATE -W -Wall -Wextra)
target_link_libraries(
    rocpd-api-stale-rocpd-decoy
    PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd-shared-library ${CMAKE_DL_LIBS})
set_target_properties(
    rocpd-api-stale-rocpd-decoy
    PROPERTIES CXX_STANDARD 17
               CXX_EXTENSIONS OFF
               CXX_STANDARD_REQUIRED ON
               LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/stale-rocpd-prefix/lib")

rocprofiler_add_integration_execute_test(
    rocpd-api-stale-install-schema-path-test
    COMMAND $<TARGET_FILE:rocpd-api-test> --all
    DEPENDS rocpd-api-test rocpd-api-stale-rocpd-decoy
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    PRELOAD "$<TARGET_FILE:rocpd-api-stale-rocpd-decoy>"
    ENVIRONMENT "ROCPD_SCHEMA_PATH=")

################################################################################
# Optional: pytest-based schema validation (same API via Python rocpd module)
################################################################################

rocprofiler_add_integration_execute_test(
    rocpd-api-python-interface-test
    COMMAND ${Python3_EXECUTABLE} -c "pass"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 10
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocpd-api-env}"
    FIXTURES_SETUP rocpd-api-python-interface)

rocprofiler_add_integration_validate_test(
    rocpd-api-python-interface-test
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "ROCPD_SCHEMA_PATH=${rocpd-api-schema-path}"
    PYTHON_PATH "${rocpd-api-site-packages}"
    FIXTURES_REQUIRED rocpd-api-python-interface
    DEPENDS rocprofiler-sdk::rocprofv3)
