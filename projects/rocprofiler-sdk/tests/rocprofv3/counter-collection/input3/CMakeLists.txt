#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-input3
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json input.yml)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")
# pmc1
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-json-pmc1
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o
        out_json -f csv -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-json-pmc1)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-json-pmc1
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_json_counter_collection.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_json_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-json-pmc1)

# YAML input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-yaml-pmc1
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o out_yaml -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-yaml-pmc1)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-yaml-pmc1
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_yaml_counter_collection.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_yaml_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-yaml-pmc1)
