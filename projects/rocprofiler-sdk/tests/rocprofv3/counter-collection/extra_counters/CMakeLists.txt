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

rocprofiler_configure_pytest_files(
    COPY input.txt extra_counters.yaml duplicate_identical.yaml run_invalid_yaml_test.py
         invalid_malformed.yaml invalid_missing_description.yaml
         invalid_mixed_definition.yaml invalid_conflicting_duplicate.yaml)

# Generate a valid definition file that exceeds Linux's 128 KiB per-environment-string
# limit. Keeping the padding in the build tree avoids committing a large fixture.
set(LARGE_EXTRA_COUNTERS_FILE "${CMAKE_CURRENT_BINARY_DIR}/large extra counters.yaml")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/extra_counters.yaml" EXTRA_COUNTERS_YAML)
string(REPEAT "# extra counter transport padding\n" 9000 EXTRA_COUNTERS_PADDING)
file(WRITE "${LARGE_EXTRA_COUNTERS_FILE}"
     "${EXTRA_COUNTERS_YAML}\n${EXTRA_COUNTERS_PADDING}")
file(SIZE "${LARGE_EXTRA_COUNTERS_FILE}" LARGE_EXTRA_COUNTERS_SIZE)
if(LARGE_EXTRA_COUNTERS_SIZE LESS 262144)
    message(FATAL_ERROR "Large extra-counter fixture must be at least 256 KiB")
endif()

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
    rocprofv3-test-counter-collection-large-extra-counters
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -E "${LARGE_EXTRA_COUNTERS_FILE}" -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_large_extra -o large
        --output-format csv -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 60
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-large-extra-counters)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-large-extra-counters
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --input
         ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_large_extra/pmc_1/large_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-large-extra-counters)

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
function(rocprofiler_add_invalid_yaml_test NAME ERROR_REGEX)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-extra-counters-invalid-${NAME}
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/run_invalid_yaml_test.py
            --timeout 15 --expect "${ERROR_REGEX}" --
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
            ${CMAKE_CURRENT_BINARY_DIR}/input.txt -E
            ${CMAKE_CURRENT_BINARY_DIR}/invalid_${NAME}.yaml -- $<TARGET_FILE:vector-ops>
        DEPENDS vector-ops
        TIMEOUT 20
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}")
endfunction()

rocprofiler_add_invalid_yaml_test(malformed "Failed to parse extra counters YAML")
rocprofiler_add_invalid_yaml_test(
    missing_description
    "Invalid extra counters YAML: Counter 'TEST': missing 'description'")
rocprofiler_add_invalid_yaml_test(
    mixed_definition "Invalid extra counters YAML:.*'expression' cannot be combined")
rocprofiler_add_invalid_yaml_test(conflicting_duplicate
                                  "Conflicting counter definitions for 'GRBM_GUI_ACTIVE'")

set(MISSING_EXTRA_COUNTERS_FILE "${CMAKE_CURRENT_BINARY_DIR}/missing-extra-counters.yaml")
rocprofiler_add_integration_execute_test(
    rocprofv3-test-extra-counters-missing-runtime-file
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/run_invalid_yaml_test.py
        --timeout 15 --expect
        "Failed to access extra counter definition file '${MISSING_EXTRA_COUNTERS_FILE}':.*No such file or directory"
        -- $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace -- ${CMAKE_COMMAND} -E
        true
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 20
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPROF_EXTRA_COUNTERS_FILE=${MISSING_EXTRA_COUNTERS_FILE}")

# Verify that each MPI rank can inherit the path and load the large definition file.
find_package(MPI QUIET)
if(NOT MPI_FOUND OR NOT MPIEXEC_EXECUTABLE)
    set(EXTRA_COUNTERS_MPI_DISABLED ON)
else()
    set(EXTRA_COUNTERS_MPI_DISABLED OFF)
endif()
set(EXTRA_COUNTERS_MPI_ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-large-extra-counters-with-mpi
    COMMAND
        ${CMAKE_COMMAND} -E env ${EXTRA_COUNTERS_MPI_ENV} ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace -E
        "${LARGE_EXTRA_COUNTERS_FILE}" -d
        ${CMAKE_CURRENT_BINARY_DIR}/large-extra-mpi/rank.%rank% -o out --output-format csv
        -- ${CMAKE_COMMAND} -E true ${MPIEXEC_POSTFLAGS}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 60
    LABELS "integration-tests" "mpi-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED ${EXTRA_COUNTERS_MPI_DISABLED})
