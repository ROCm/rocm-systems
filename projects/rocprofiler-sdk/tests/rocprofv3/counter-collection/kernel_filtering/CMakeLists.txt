#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-kernel-filtering
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.txt input.json input.yml)

# pmc1
add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-input-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/json_input
        -- $<TARGET_FILE:vector-ops>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-kernel-filtering "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-input-json-execute
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" ENVIRONMENT
               "${cc-env-kernel-filtering}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-input-cmd-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt --kernel-include-regex ".*kernel"
        --kernel-exclude-regex "multiply" -T -d ${CMAKE_CURRENT_BINARY_DIR}/cmd_input -o
        out --output-format csv -- $<TARGET_FILE:vector-ops>)

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-input-cmd-execute
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" ENVIRONMENT
               "${cc-env-kernel-filtering}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# pmc1
add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-input-yaml-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.yml -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/yaml_input -o out --output-format csv json --
        $<TARGET_FILE:vector-ops>)

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-input-yaml-execute
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" ENVIRONMENT
               "${cc-env-kernel-filtering}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-input-json-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k _pass
        --input-csv-pass1
        ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_1/out_counter_collection.csv
        --input-json-pass1 ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_1/out_results.json
        --input-csv-pass2
        ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_2/out_counter_collection.csv
        --input-json-pass2 ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_2/out_results.json
        --input-csv-pass3
        ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_3/out_counter_collection.csv
        --input-json-pass3 ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_3/out_results.json
        --input-csv-pass4
        ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_4/out_counter_collection.csv
        --input-json-pass4 ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_4/out_results.json)

add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-input-yaml-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k _pass
        --input-csv-pass1
        ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_1/out_counter_collection.csv
        --input-json-pass1 ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_1/out_results.json
        --input-csv-pass2
        ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_2/out_counter_collection.csv
        --input-json-pass2 ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_2/out_results.json
        --input-csv-pass3
        ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_3/out_counter_collection.csv
        --input-json-pass3 ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_3/out_results.json
        --input-csv-pass4
        ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_4/out_counter_collection.csv
        --input-json-pass4 ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_4/out_results.json)

add_test(
    NAME rocprofv3-test-counter-collection-kernel-filtering-input-cmd-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        test_validate_counter_collection_csv_pmc1 --input-csv-pmc1
        ${CMAKE_CURRENT_BINARY_DIR}/cmd_input/pmc_1/out_counter_collection.csv)

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-input-json-validate
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-counter-collection-kernel-filtering-input-json-execute"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-input-cmd-validate
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-counter-collection-kernel-filtering-input-cmd-execute"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set_tests_properties(
    rocprofv3-test-counter-collection-kernel-filtering-input-yaml-validate
    PROPERTIES TIMEOUT 120 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-counter-collection-kernel-filtering-input-yaml-execute"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
