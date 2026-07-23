#
# rocprofv3 counter collection option validation tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-invalid-options
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-invalid-options
    TEST_PATHS validate.py
    COPY conftest.py pmc_groups.json
    CONFIG pytest.ini
    ARGS --rocprofv3 $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc-groups-input
         ${CMAKE_CURRENT_BINARY_DIR}/pmc_groups.json
    TIMEOUT 45
    LABELS "integration-tests")
