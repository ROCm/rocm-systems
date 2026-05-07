#
# rocprofv3 tool test - SQG-block counter collection on gfx11
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-sqg-counters
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.txt)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# Run vector-ops with the multi-pass SQG counter input. The SQG block on gfx11 has 8
# hardware counter slots, so input.txt schedules 5 passes (8+8+8+8+4 = 36 distinct SQG
# events) covering every newly-added SQG counter.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-sqg-counters
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        -T -d ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters -o sqg --output-format csv json
        -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-sqg-counters)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-sqg-counters
    TEST_PATHS validate.py
    COPY conftest.py input.txt
    CONFIG pytest.ini
    ARGS --input-csv-pmc1
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_1/sqg_counter_collection.csv
         --input-json-pmc1
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_1/sqg_results.json
         --input-csv-pmc2
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_2/sqg_counter_collection.csv
         --input-json-pmc2
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_2/sqg_results.json
         --input-csv-pmc3
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_3/sqg_counter_collection.csv
         --input-json-pmc3
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_3/sqg_results.json
         --input-csv-pmc4
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_4/sqg_counter_collection.csv
         --input-json-pmc4
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_4/sqg_results.json
         --input-csv-pmc5
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_5/sqg_counter_collection.csv
         --input-json-pmc5
         ${CMAKE_CURRENT_BINARY_DIR}/out_sqg_counters/pmc_5/sqg_results.json
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-sqg-counters)
