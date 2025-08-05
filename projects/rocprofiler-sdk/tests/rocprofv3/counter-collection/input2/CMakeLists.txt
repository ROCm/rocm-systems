#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-input2
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.txt)

# pmc2
add_test(
    NAME rocprofv3-test-counter-collection-txt-pmc2-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt --output-format CSV JSON -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o out -- $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-pmc2 "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-counter-collection-txt-pmc2-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc2}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-txt-pmc2-execute-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_1/out_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_1/out_counter_collection.csv
        --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_2/out_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/pmc_2/out_counter_collection.csv)

set_tests_properties(
    rocprofv3-test-counter-collection-txt-pmc2-execute-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-counter-collection-pmc2-execute" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
