#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-range-filtering
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input_range.json)

add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-range-filter-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_range.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/range_filter -- $<TARGET_FILE:transpose> 1 15)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-kernel-filtering "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-range-filter-execute
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" ENVIRONMENT
               "${cc-env-kernel-filtering}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-range-filter-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-config
        ${CMAKE_CURRENT_BINARY_DIR}/input_range.json --input-json-pass1
        ${CMAKE_CURRENT_BINARY_DIR}/range_filter/pass_1/out_results.json
        --input-json-pass2
        ${CMAKE_CURRENT_BINARY_DIR}/range_filter/pass_2/out_results.json
        --input-json-pass3
        ${CMAKE_CURRENT_BINARY_DIR}/range_filter/pass_3/out_results.json)

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-range-filter-validate
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-counter-collection-kernel-filtering-range-filter-execute"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
