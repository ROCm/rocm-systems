#
# rocprofv3 tool
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py input.txt
                                   CONFIG pytest.ini)

# pmc3
add_test(
    NAME rocprofv3-test-tracing-plus-cc-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -d ${CMAKE_CURRENT_BINARY_DIR}/out_cc_trace
        -o pmc3 --output-format JSON:CSV $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-tracing-env "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-tracing-plus-cc-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests;application-replay" ENVIRONMENT
               "${cc-tracing-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

foreach(_DIR "pmc_1" "pmc_2" "pmc_3" "pmc_4")
    add_test(
        NAME rocprofv3-test-tracing-plus-cc-validate-${_DIR}
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_cc_trace/${_DIR}/pmc3_results.json"
            --hsa-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_cc_trace/${_DIR}/pmc3_hsa_api_trace.csv"
            --agent-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_cc_trace/${_DIR}/pmc3_agent_info.csv"
            --counter-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_cc_trace/${_DIR}/pmc3_counter_collection.csv"
        )

    set_tests_properties(
        rocprofv3-test-tracing-plus-cc-validate-${_DIR}
        PROPERTIES TIMEOUT 45 LABELS "integration-tests;application-replay" DEPENDS
                   "rocprofv3-test-tracing-plus-cc-execute" FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()
