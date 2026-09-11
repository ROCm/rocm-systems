#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-list-metrics
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(INVALID_YAML_CASES malformed missing_description mixed_definition
                       conflicting_duplicate)
set(INVALID_YAML_FILES)
foreach(invalid_yaml IN LISTS INVALID_YAML_CASES)
    list(APPEND INVALID_YAML_FILES invalid_${invalid_yaml}.yaml)
endforeach()

rocprofiler_configure_pytest_files(
    COPY input.txt extra_counters.yaml duplicate_identical.yaml run_invalid_yaml_test.py
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

rocprofiler_add_integration_execute_test(
    rocprofv3-test-extra-counters-identical-duplicate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E ${CMAKE_CURRENT_BINARY_DIR}/duplicate_identical.yaml -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_identical_duplicate -o
        duplicate --output-format csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION
        "Counter 'TEST_YAML_LOAD'.*duplicates an identical definition; ignoring duplicate"
    FIXTURES_SETUP rocprofv3-test-extra-counters-identical-duplicate)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-extra-counters-identical-duplicate
    TEST_PATHS validate.py
    CONFIG pytest.ini
    ARGS --input
         ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_identical_duplicate/pmc_1/duplicate_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-extra-counters-identical-duplicate)

# Representative parser, schema, and loader failures must produce a specific fatal
# diagnostic without hanging. Exhaustive schema cases are covered by unit tests.
set(malformed_error_regex "Failed to parse extra counters YAML")
set(missing_description_error_regex
    "Invalid extra counters YAML: Counter 'TEST': missing 'description'")
set(mixed_definition_error_regex
    "Invalid extra counters YAML:.*'expression' cannot be combined")
set(conflicting_duplicate_error_regex
    "Conflicting counter definitions for 'GRBM_GUI_ACTIVE'")

foreach(invalid_yaml IN LISTS INVALID_YAML_CASES)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-extra-counters-invalid-${invalid_yaml}
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/run_invalid_yaml_test.py
            --timeout 15 --expect "${${invalid_yaml}_error_regex}" --
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
            ${CMAKE_CURRENT_BINARY_DIR}/input.txt -E
            ${CMAKE_CURRENT_BINARY_DIR}/invalid_${invalid_yaml}.yaml --
            $<TARGET_FILE:vector-ops>
        DEPENDS vector-ops
        TIMEOUT 20
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}")
endforeach()
