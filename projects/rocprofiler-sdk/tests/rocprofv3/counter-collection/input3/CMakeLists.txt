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

# pmc1
add_test(
    NAME rocprofv3-test-counter-collection-json-pmc1-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc
        -o out_json -f csv -- $<TARGET_FILE:simple-transpose>)

add_test(
    NAME rocprofv3-test-counter-collection-yaml-pmc1-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.yml -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o
        out_yaml -f csv -- $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-pmc1 "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-counter-collection-json-pmc1-execute
    rocprofv3-test-counter-collection-yaml-pmc1-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc1}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-json-pmc1-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_json_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_json_counter_collection.csv
        --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_json_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_json_counter_collection.csv
    )

set_tests_properties(
    rocprofv3-test-counter-collection-json-pmc1-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-counter-collection-json-pmc1-execute
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-yaml-pmc1-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_yaml_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_1/out_yaml_counter_collection.csv
        --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_yaml_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pass_2/out_yaml_counter_collection.csv
    )

set_tests_properties(
    rocprofv3-test-counter-collection-yaml-pmc1-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-counter-collection-yaml-pmc1-execute
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
