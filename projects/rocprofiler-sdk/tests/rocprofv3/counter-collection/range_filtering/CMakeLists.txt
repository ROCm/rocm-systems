#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-range-filtering
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# Range filtering test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-range-filter
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_range.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/range_filter -- $<TARGET_FILE:transpose> 1 15
    DEPENDS transpose
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-range-filter)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-range-filter
    TEST_PATHS validate.py
    COPY conftest.py input_range.json
    CONFIG pytest.ini
    ARGS --json-config
         ${CMAKE_CURRENT_BINARY_DIR}/input_range.json
         --input-json-pass1
         ${CMAKE_CURRENT_BINARY_DIR}/range_filter/pass_1/out_results.json
         --input-json-pass2
         ${CMAKE_CURRENT_BINARY_DIR}/range_filter/pass_2/out_results.json
         --input-json-pass3
         ${CMAKE_CURRENT_BINARY_DIR}/range_filter/pass_3/out_results.json
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-range-filter)
