#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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
         --iteration-config
         ${CMAKE_CURRENT_BINARY_DIR}/input.json
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

# iteration-range on the command line; defined once so the CLI and validator args match
set(_ITER_RANGE "[1-2]")

# --kernel-trace exposes the full (unfiltered) kernel_dispatch trace so the validator can
# prove which iterations were captured
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-cmd-iteration-range
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        --kernel-include-regex ".*_kernel" --kernel-iteration-range "${_ITER_RANGE}"
        --kernel-trace -T -d ${CMAKE_CURRENT_BINARY_DIR}/cmd_iteration_range_input -o out
        --output-format csv json -- $<TARGET_FILE:vector-ops> 3 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP
        rocprofv3-test-counter-collection-kernel-filtering-input-cmd-iteration-range)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-input-cmd-iteration-range
    TEST_PATHS validate.py
    COPY conftest.py input.txt input.json input.yml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k iteration_range
    ARGS --input-csv-iteration-range
         ${CMAKE_CURRENT_BINARY_DIR}/cmd_iteration_range_input/pmc_1/out_counter_collection.csv
         --input-json-iteration-range
         ${CMAKE_CURRENT_BINARY_DIR}/cmd_iteration_range_input/pmc_1/out_results.json
         --kernel-iteration-range
         "${_ITER_RANGE}"
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        rocprofv3-test-counter-collection-kernel-filtering-input-cmd-iteration-range)

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
         --iteration-config
         ${CMAKE_CURRENT_BINARY_DIR}/input.yml
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-input-yaml)

# CLI filter coverage: --kernel-include-regex / --kernel-exclude-regex /
# --kernel-iteration-range on the command line, plus edge cases.

# defined once so the CLI argument and the validator argument cannot drift
set(_CLI_ITER_RANGE "[1-2]")

# single included kernel + iteration range; --kernel-trace lets the validator recover the
# exact captured iterations
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-single-kernel
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex
        "addition_kernel" --kernel-iteration-range "${_CLI_ITER_RANGE}" --kernel-trace -T
        -d ${CMAKE_CURRENT_BINARY_DIR}/cli_single_kernel -o out --output-format csv json
        -- $<TARGET_FILE:vector-ops> 8 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-single-kernel)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-single-kernel
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k cli_single_kernel
    ARGS --input-csv-file
         ${CMAKE_CURRENT_BINARY_DIR}/cli_single_kernel/out_counter_collection.csv
         --input-json-file
         ${CMAKE_CURRENT_BINARY_DIR}/cli_single_kernel/out_results.json
         --kernel-iteration-range
         "${_CLI_ITER_RANGE}"
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-cli-single-kernel
    )

# multi-token range ("1" "2") must produce the same result as "[1-2]"
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-multitoken-range
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex
        "addition_kernel" --kernel-iteration-range "1" "2" --kernel-trace -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/cli_multitoken_range -o out --output-format csv json
        -- $<TARGET_FILE:vector-ops> 8 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-multitoken-range
    )

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-multitoken-range
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k cli_single_kernel
    ARGS --input-csv-file
         ${CMAKE_CURRENT_BINARY_DIR}/cli_multitoken_range/out_counter_collection.csv
         --input-json-file
         ${CMAKE_CURRENT_BINARY_DIR}/cli_multitoken_range/out_results.json
         --kernel-iteration-range
         "${_CLI_ITER_RANGE}"
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        rocprofv3-test-counter-collection-kernel-filtering-cli-multitoken-range)

# exclude one kernel from the included set
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-exclude
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex
        ".*_kernel" --kernel-exclude-regex "multiply" -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/cli_exclude -o out --output-format csv --
        $<TARGET_FILE:vector-ops> 8 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-exclude)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-exclude
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k cli_exclude_csv
    ARGS --input-csv-file
         ${CMAKE_CURRENT_BINARY_DIR}/cli_exclude/out_counter_collection.csv
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-cli-exclude)

# --mangled-kernels: regex matches the mangled symbol
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-mangled
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex
        "addition_kernel" -M -d ${CMAKE_CURRENT_BINARY_DIR}/cli_mangled -o out
        --output-format csv -- $<TARGET_FILE:vector-ops> 8 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-mangled)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-mangled
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k cli_mangled_csv
    ARGS --input-csv-file
         ${CMAKE_CURRENT_BINARY_DIR}/cli_mangled/out_counter_collection.csv
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-cli-mangled)

# runs that complete successfully with zero counter records: exclude cancels the include
# set, no-match include regex, and an out-of-bounds range (--kernel-trace forces an output
# file to validate against)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-exclude-all
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex
        ".*_kernel" --kernel-exclude-regex ".*_kernel" --kernel-trace -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/cli_exclude_all -o out --output-format json --
        $<TARGET_FILE:vector-ops> 8 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-exclude-all)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-exclude-all
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k cli_empty_result_json
    ARGS --input-json-file ${CMAKE_CURRENT_BINARY_DIR}/cli_exclude_all/out_results.json
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-cli-exclude-all)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-empty-match
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex
        "no_such_kernel" --kernel-trace -T -d ${CMAKE_CURRENT_BINARY_DIR}/cli_empty_match
        -o out --output-format json -- $<TARGET_FILE:vector-ops> 8 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-empty-match)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-empty-match
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k cli_empty_result_json
    ARGS --input-json-file ${CMAKE_CURRENT_BINARY_DIR}/cli_empty_match/out_results.json
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-kernel-filtering-cli-empty-match)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-range-out-of-bounds
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex
        ".*_kernel" --kernel-iteration-range "[99]" --kernel-trace -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/cli_range_out_of_bounds -o out --output-format json --
        $<TARGET_FILE:vector-ops> 8 1
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP
        rocprofv3-test-counter-collection-kernel-filtering-cli-range-out-of-bounds)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-range-out-of-bounds
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k cli_empty_result_json
    ARGS --input-json-file
         ${CMAKE_CURRENT_BINARY_DIR}/cli_range_out_of_bounds/out_results.json
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        rocprofv3-test-counter-collection-kernel-filtering-cli-range-out-of-bounds)

# malformed / non-integer ranges must abort with a fatal message. The abort (SIGABRT)
# matches ROCPROFILER_DEFAULT_FAIL_REGEX, so the run is piped to grep and grep's status
# becomes the result. That asserts the diagnostic is emitted, which only happens on the
# fatal path; the tool's own exit status is consumed by the pipeline and not checked.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-malformed-range
    COMMAND
        sh -c
        "$<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex '.*_kernel' --kernel-iteration-range '[1-]' -T -d ${CMAKE_CURRENT_BINARY_DIR}/cli_malformed_range -o out --output-format json -- $<TARGET_FILE:vector-ops> 8 1 2>&1 | grep 'bad range format'"
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-malformed-range)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-kernel-filtering-cli-noninteger-range
    COMMAND
        sh -c
        "$<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --kernel-include-regex '.*_kernel' --kernel-iteration-range '[a]' -T -d ${CMAKE_CURRENT_BINARY_DIR}/cli_noninteger_range -o out --output-format json -- $<TARGET_FILE:vector-ops> 8 1 2>&1 | grep 'Non-integer value detected'"
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-kernel-filtering-cli-noninteger-range
    )
