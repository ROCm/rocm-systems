#
# rocpd API tests: C++ rocpd-api-test app (main) plus optional pytest schema checks
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

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
