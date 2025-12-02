#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-minimum-bytes
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# pmc2
rocprofiler_add_integration_execute_test(
    rocprofv3-test-minimum-bytes
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json --output-format csv json pftrace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-kernel-trace -o out --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-minimum-bytes)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-minimum-bytes
    TEST_PATHS validate.py
    COPY conftest.py input.json
    CONFIG pytest.ini
    ARGS --trace-input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-kernel-trace/out*.csv
         --trace-input-json
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-kernel-trace/out_results.json
         --trace-input-pftrace
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-kernel-trace/out_results.pftrace
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-minimum-bytes)
