#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-list-metrics
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(INVALID_YAML_CASES
    malformed
    missing_root
    missing_counters
    missing_name
    empty_arch
    no_event_expr
    event_no_block
    block_no_event
    conflicting_duplicate)
set(INVALID_YAML_FILES)
foreach(invalid_yaml IN LISTS INVALID_YAML_CASES)
    list(APPEND INVALID_YAML_FILES invalid_${invalid_yaml}.yaml)
endforeach()

rocprofiler_configure_pytest_files(COPY input.txt extra_counters.yaml
                                        ${INVALID_YAML_FILES})

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
    PASS_REGULAR_EXPRESSION
        "Counter 'TEST_YAML_LOAD'.*duplicates an identical definition; ignoring duplicate"
    FIXTURES_SETUP rocprofv3-test-counter-collection-txt-pmc1-extra-counters)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-pmc1-extra-counters
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --input
         ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_1_extra/pmc_1/pmc1_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-txt-pmc1-extra-counters)

# Invalid YAML must produce a field-specific fatal diagnostic without hanging.
set(malformed_error_regex "Failed to parse extra counters YAML")
set(missing_root_error_regex "Invalid extra counters YAML: Missing top-level")
set(missing_counters_error_regex "Invalid extra counters YAML: Missing 'counters' array")
set(missing_name_error_regex
    "Invalid extra counters YAML: counters.0.: missing 'name' field")
set(empty_arch_error_regex "Invalid extra counters YAML:.*'architectures' is empty")
set(no_event_expr_error_regex "Invalid extra counters YAML:.*must have 'expression' or")
set(event_no_block_error_regex "Invalid extra counters YAML:.*'event' requires 'block'")
set(block_no_event_error_regex "Invalid extra counters YAML:.*'block' requires 'event'")
set(conflicting_duplicate_error_regex
    "Conflicting counter definitions for 'GRBM_GUI_ACTIVE' on architecture "
    "'gfx942'.*append mode cannot override an existing counter")

foreach(invalid_yaml IN LISTS INVALID_YAML_CASES)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-extra-counters-invalid-${invalid_yaml}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
            ${CMAKE_CURRENT_BINARY_DIR}/input.txt -E
            ${CMAKE_CURRENT_BINARY_DIR}/invalid_${invalid_yaml}.yaml --
            $<TARGET_FILE:vector-ops>
        DEPENDS vector-ops
        TIMEOUT 10
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT "ROCPROF_INTERNAL_TEST_SIGNAL_HANDLER_VIA_EXIT=1"
        PASS_REGULAR_EXPRESSION "${${invalid_yaml}_error_regex}")
endforeach()
