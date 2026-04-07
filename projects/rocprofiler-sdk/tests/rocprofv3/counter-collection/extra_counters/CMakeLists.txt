#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-extra-counters
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")
# pmc1 with extra counters
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-txt-pmc1-extra-counters
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/extra_counters.yaml -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_1_extra -o pmc1 --output-format
        csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-txt-pmc1-extra-counters)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-pmc1-extra-counters
    TEST_PATHS validate.py
    COPY conftest.py input.txt extra_counters.yaml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k test_validate_counter_collection_pmc1_extra_counters
    ARGS --input
         ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_1_extra/pmc_1/pmc1_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-txt-pmc1-extra-counters)

#
# YAML Validation Tests - Invalid YAMLs should be handled gracefully without crashes
#
rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-duplicate-counters
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/duplicate_counters.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_duplicate -o duplicate_test --output-format csv --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-invalid-block
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/invalid_block.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_invalid_block -o invalid_block_test
        --output-format csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-missing-name
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/missing_name.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_missing_name -o missing_name_test --output-format
        csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-wrong-type
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/wrong_type.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_wrong_type -o wrong_type_test --output-format csv
        -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-malformed
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/malformed.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_malformed -o malformed_test --output-format csv --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-no-block-event
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/no_block_event.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_no_block_event -o no_block_event_test
        --output-format csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-missing-top-key
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/missing_top_key.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_missing_top_key -o missing_top_key_test
        --output-format csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-yaml-validation-empty-architectures
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/empty_architectures.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_empty_architectures -o empty_architectures_test
        --output-format csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")
