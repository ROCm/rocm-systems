#
# rocprofv3 tool
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-tracing-plus-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py input.txt
                                   CONFIG pytest.ini)

# pmc
add_test(
    NAME rocprofv3-test-tracing-plus-counter-collection-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace --kernel-trace -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace -o pmc --output-format
        JSON PFTRACE CSV -- $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-tracing-env "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-tracing-plus-counter-collection-execute
    PROPERTIES TIMEOUT 90 LABELS "integration-tests;application-replay" ENVIRONMENT
               "${cc-tracing-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

foreach(_DIR "pmc_1" "pmc_2" "pmc_3" "pmc_4")
    add_test(
        NAME rocprofv3-test-tracing-plus-counter-collection-validate-${_DIR}
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_results.json"
            --pftrace-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_results.pftrace"
            --hsa-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_hsa_api_trace.csv"
            --agent-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_agent_info.csv"
            --counter-input
            "${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_counter_collection.csv"
        )

    set_tests_properties(
        rocprofv3-test-tracing-plus-counter-collection-validate-${_DIR}
        PROPERTIES TIMEOUT 45 LABELS "integration-tests;application-replay" DEPENDS
                   "rocprofv3-test-tracing-plus-counter-collection-execute"
                   FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()

set(cmdl_single "GRBM_COUNT")
set(cmdl_multiple "SQ_WAVES" "GRBM_GUI_ACTIVE" "SQ_CYCLES:device=16384"
                  "SQ_BUSY_CYCLES:device=65536")

foreach(_GROUP "single" "multiple")

    set(_TEST_NAME_PREFIX rocprofv3-test-tracing-plus-counter-collection-cmdl-${_GROUP})

    add_test(
        NAME ${_TEST_NAME_PREFIX}-execute
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace --kernel-rename --pmc
            ${cmdl_${_GROUP}} -d ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace
            -o cmdl/${_GROUP} --output-format JSON PFTRACE CSV --
            $<TARGET_FILE:transpose> 2 50 5)

    set_tests_properties(
        ${_TEST_NAME_PREFIX}-execute
        PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-tracing-env}"
                   FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

    set(_TEST_OUTPUT_PREFIX
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/cmdl/${_GROUP})

    add_test(
        NAME ${_TEST_NAME_PREFIX}-validate
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
            "${_TEST_OUTPUT_PREFIX}_results.json" --pftrace-input
            "${_TEST_OUTPUT_PREFIX}_results.pftrace" --hsa-input
            "${_TEST_OUTPUT_PREFIX}_hsa_api_trace.csv" --agent-input
            "${_TEST_OUTPUT_PREFIX}_agent_info.csv" --counter-input
            "${_TEST_OUTPUT_PREFIX}_counter_collection.csv")

    set_tests_properties(
        ${_TEST_NAME_PREFIX}-validate
        PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
                   "${_TEST_NAME_PREFIX}-execute" FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()
