#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-kernel-filtering
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")
# pmc1
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/json_input
        -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-input-json)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-json
    TEST_PATHS validate.py
    COPY conftest.py input.txt input.json input.yml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k _pass
    ARGS --input-csv-pass1
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_1/out_counter_collection.csv
         --input-json-pass1
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_1/out_results.json
         --input-csv-pass2
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_2/out_counter_collection.csv
         --input-json-pass2
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_2/out_results.json
         --input-csv-pass3
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_3/out_counter_collection.csv
         --input-json-pass3
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_3/out_results.json
         --input-csv-pass4
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_4/out_counter_collection.csv
         --input-json-pass4
         ${CMAKE_CURRENT_BINARY_DIR}/json_input/pass_4/out_results.json
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-input-json)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-cmd
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        --kernel-include-regex ".*kernel" --kernel-exclude-regex "multiply" -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/cmd_input -o out --output-format csv --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-input-cmd)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-cmd
    TEST_PATHS validate.py
    COPY conftest.py input.txt input.json input.yml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k test_validate_counter_collection_csv_pmc1
    ARGS --input-csv-pmc1
         ${CMAKE_CURRENT_BINARY_DIR}/cmd_input/pmc_1/out_counter_collection.csv
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-input-cmd)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -T -d ${CMAKE_CURRENT_BINARY_DIR}/yaml_input -o out --output-format csv json --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-input-yaml)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.txt input.json input.yml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k _pass
    ARGS --input-csv-pass1
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_1/out_counter_collection.csv
         --input-json-pass1
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_1/out_results.json
         --input-csv-pass2
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_2/out_counter_collection.csv
         --input-json-pass2
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_2/out_results.json
         --input-csv-pass3
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_3/out_counter_collection.csv
         --input-json-pass3
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_3/out_results.json
         --input-csv-pass4
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_4/out_counter_collection.csv
         --input-json-pass4
         ${CMAKE_CURRENT_BINARY_DIR}/yaml_input/pass_4/out_results.json
    TIMEOUT 120
    LABELS "integration-tests" FIXTURES
           rocprofv3-test-counter-collection-kernel-filtering-input-yaml)
