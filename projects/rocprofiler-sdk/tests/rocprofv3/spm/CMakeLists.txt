#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-spm
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

set(COUNTER_LIST
    SQ_CYCLES
    SQ_WAVES
    SQ_INSTS_VALU
    SQ_INSTS_SALU
    SQ_INSTS_SALU
    TCC_HIT
    TCC_MISS
    TCC_REQ
    CPC_CPC_STAT_BUSY
    CPC_CPC_STAT_IDLE
    SPI_CSN_NUM_THREADGROUPS)

set(IS_DISABLED TRUE)

list(GET rocprofiler-sdk-tests-gfx-info 0 spm-gpu-0-gfx-info)

# SPM
add_test(
    NAME rocprofv3-test-spm-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm ${COUNTER_LIST} --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format CSV JSON -o spm --
        $<TARGET_FILE:simple-transpose>)

# PMC
add_test(
    NAME rocprofv3-test-spm-compare-to-pmc
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc ${COUNTER_LIST} -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format CSV -o pmc --
        $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-spm-execute
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})

set_tests_properties(
    rocprofv3-test-spm-compare-to-pmc
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})

add_test(
    NAME rocprofv3-test-spm-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --pmc
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm/pmc_counter_collection.csv --spm
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_streaming_performance_monitor.csv
        --spm-json ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.json --gfx-info
        ${spm-gpu-0-gfx-info})

set_tests_properties(
    rocprofv3-test-spm-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-spm-execute;rocprofv3-test-spm-compare-to-pmc"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})

# Multi-gpu + multiqueue
add_test(
    NAME rocprofv3-test-spm-multigpu
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm ${COUNTER_LIST}
        --spm-frequency-sclk 800000 --spm-timeout-ms 10 --spm-buffer-size 4096 -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_multi --output-format CSV -o multi --
        $<TARGET_FILE:vector-ops>)

set_tests_properties(
    rocprofv3-test-spm-multigpu
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})
