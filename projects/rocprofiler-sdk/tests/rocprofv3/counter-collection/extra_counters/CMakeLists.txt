#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-list-metrics
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.txt extra_counters.yaml)

# pmc1 with extra counters
add_test(
    NAME rocprofv3-test-counter-collection-txt-pmc1-extra-counters-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -E
        ${CMAKE_CURRENT_BINARY_DIR}/extra_counters.yaml -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_1_extra -o pmc1
        --output-format csv -- $<TARGET_FILE:vector-ops>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-pmc1 "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-counter-collection-txt-pmc1-extra-counters-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc1}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-pmc1-extra-counters-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_1_extra/pmc_1/pmc1_counter_collection.csv
    )

set_tests_properties(
    rocprofv3-test-counter-collection-pmc1-extra-counters-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-counter-collection-txt-pmc1-extra-counters-execute"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
