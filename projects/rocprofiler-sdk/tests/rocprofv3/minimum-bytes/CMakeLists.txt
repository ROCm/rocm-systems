#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-minimum-bytes
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json)

# pmc2
add_test(
    NAME rocprofv3-test-minimum-bytes-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json --output-format csv json pftrace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-kernel-trace -o out --
        $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-pmc2 "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-minimum-bytes-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc2}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-minimum-bytes-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --trace-input-csv
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-kernel-trace/out*.csv
        --trace-input-json
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-kernel-trace/out_results.json
        --trace-input-pftrace
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-kernel-trace/out_results.pftrace)

set_tests_properties(
    rocprofv3-test-minimum-bytes-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-minimum-bytes-execute" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
