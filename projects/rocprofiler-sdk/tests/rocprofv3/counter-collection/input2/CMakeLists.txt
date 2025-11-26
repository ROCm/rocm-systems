#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-input2
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# pmc2
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-txt-pmc2
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.txt
        --output-format CSV JSON -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o out --
        $<TARGET_FILE:simple-transpose>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-txt-pmc2)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-txt-pmc2
    TEST_PATHS validate.py
    COPY conftest.py input.txt
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_1/out_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_1/out_counter_collection.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_2/out_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_2/out_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-txt-pmc2)
