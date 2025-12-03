#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-spm
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(COUNTER_LIST
    SQ_CYCLES
    SQ_WAVES
    SQ_INSTS_VALU
    SQ_INSTS_SALU
    TCC_HIT
    TCC_MISS
    TCC_REQ
    CPC_CPC_STAT_BUSY
    CPC_CPC_STAT_IDLE)

rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
set(IS_DISABLED ${IS_SPM_DISABLED})

# SPM
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled=on --spm
        ${COUNTER_LIST} --spm-frequency 50000 --spm-buffer-size 32768 --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format CSV JSON -o spm --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# PMC
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-compare-to-pmc
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc ${COUNTER_LIST} -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format CSV -o pmc --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-validate
    TEST_PATHS validate.py
    ARGS --pmc ${CMAKE_CURRENT_BINARY_DIR}/out_spm/pmc_counter_collection.csv --spm
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_streaming_performance_monitor.csv
         --spm-json ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.json
    CONFIG pytest.ini
    COPY spm_input.json conftest.py
    LABELS "integration-tests"
    FIXTURES_REQUIRED "rocprofv3-test-spm-execute;rocprofv3-test-spm-compare-to-pmc"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# Multi-gpu + multiqueue
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-multigpu
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i spm_input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_multi --output-format CSV -o multi --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})
