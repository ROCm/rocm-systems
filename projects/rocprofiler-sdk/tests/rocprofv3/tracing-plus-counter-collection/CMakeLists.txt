#
# rocprofv3 tool
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-tracing-plus-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# WSL/DXG does not expose /dev/kfd and currently uses libhsakmt's
# vendor-packet path for counter collection. Single-counter collection works,
# but larger multi-counter AQL/PM4 packets can exceed libhsakmt's per-queue PM4
# frame-size limit until that is raised. Keep single-counter coverage enabled
# and disable only multi-counter integration paths on WSL/no-KFD.
set(IS_WSL_DXG_MULTI_COUNTER_DISABLED OFF)
if(NOT EXISTS "/dev/kfd")
    set(IS_WSL_DXG_MULTI_COUNTER_DISABLED ON)
endif()

# pmc
rocprofiler_add_integration_execute_test(
    rocprofv3-test-tracing-plus-counter-collection
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace --kernel-trace -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace -o pmc --output-format
        JSON PFTRACE CSV -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 90
    LABELS "integration-tests;application-replay"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${IS_WSL_DXG_MULTI_COUNTER_DISABLED}"
    FIXTURES_SETUP rocprofv3-test-tracing-plus-counter-collection)

foreach(_DIR "pmc_1" "pmc_2" "pmc_3" "pmc_4")
    rocprofiler_add_integration_validate_test(
        rocprofv3-test-tracing-plus-counter-collection-${_DIR}
        TEST_PATHS validate.py
        COPY conftest.py input.txt
        CONFIG pytest.ini
        ARGS --json-input
             ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_results.json
             --pftrace-input
             ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_results.pftrace
             --hsa-input
             ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_hsa_api_trace.csv
             --agent-input
             ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_agent_info.csv
             --counter-input
             ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/${_DIR}/pmc_counter_collection.csv
        TIMEOUT 45
        LABELS "integration-tests;application-replay"
        DISABLED "${IS_WSL_DXG_MULTI_COUNTER_DISABLED}"
        FIXTURES_REQUIRED rocprofv3-test-tracing-plus-counter-collection)
endforeach()

set(cmdl_single "GRBM_COUNT")
set(cmdl_multiple "SQ_WAVES" "GRBM_GUI_ACTIVE" "SQ_CYCLES:device=16384"
                  "SQ_BUSY_CYCLES:device=65536")

foreach(_GROUP "single" "multiple")
    set(_CMDL_MULTI_COUNTER_DISABLED OFF)
    if(_GROUP STREQUAL "multiple")
        set(_CMDL_MULTI_COUNTER_DISABLED "${IS_WSL_DXG_MULTI_COUNTER_DISABLED}")
    endif()

    rocprofiler_add_integration_execute_test(
        rocprofv3-test-tracing-plus-counter-collection-cmdl-${_GROUP}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace --kernel-rename --pmc
            ${cmdl_${_GROUP}} -d ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace
            -o cmdl/${_GROUP} --output-format JSON PFTRACE CSV -- $<TARGET_FILE:transpose>
            2 50 5
        DEPENDS transpose
        TIMEOUT 45
        PRELOAD "${PRELOAD_ENV}"
        DISABLED "${_CMDL_MULTI_COUNTER_DISABLED}"
        FIXTURES_SETUP rocprofv3-test-tracing-plus-counter-collection-cmdl-${_GROUP})

    set(_TEST_OUTPUT_PREFIX
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_trace/cmdl/${_GROUP})

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-tracing-plus-counter-collection-cmdl-${_GROUP}
        TEST_PATHS validate.py
        COPY conftest.py input.txt
        CONFIG pytest.ini
        ARGS --json-input
             ${_TEST_OUTPUT_PREFIX}_results.json
             --pftrace-input
             ${_TEST_OUTPUT_PREFIX}_results.pftrace
             --hsa-input
             ${_TEST_OUTPUT_PREFIX}_hsa_api_trace.csv
             --agent-input
             ${_TEST_OUTPUT_PREFIX}_agent_info.csv
             --counter-input
             ${_TEST_OUTPUT_PREFIX}_counter_collection.csv
        TIMEOUT 45
        LABELS "integration-tests"
        DISABLED "${_CMDL_MULTI_COUNTER_DISABLED}"
        FIXTURES_REQUIRED rocprofv3-test-tracing-plus-counter-collection-cmdl-${_GROUP})
endforeach()
