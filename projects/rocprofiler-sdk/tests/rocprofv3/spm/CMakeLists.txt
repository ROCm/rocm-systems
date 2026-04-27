#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-spm
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

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

if(NOT ROCPROFILER_MEMCHECK STREQUAL "")
    set(SANITIZER True)
else()
    set(SANITIZER False)
endif()

rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
if(${SANITIZER})
    set(IS_SPM_DISABLED True)
endif()
set(IS_DISABLED True)

set(rocprofv3-spm-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

# SPM
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled=on --spm
        ${COUNTER_LIST} --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1200
        --kernel-trace -d ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format json rocpd
        -o spm -- $<TARGET_FILE:simple-transpose>
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
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format json -o pmc --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# rocpd csv generation from spm database
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-rocpd-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm/rocpd_csv -i
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${rocprofv3-spm-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-spm-rocpd-csv-generation
    FIXTURES_REQUIRED rocprofv3-test-spm-execute
    DISABLED ${IS_DISABLED})

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-validate
    TEST_PATHS validate.py
    ARGS --pmc-json
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/pmc_results.json
         --spm-json
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.json
         --counter-csv
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/rocpd_csv/out_spm_counter_collection_trace.csv
    CONFIG pytest.ini
    COPY spm_input.json conftest.py
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        "rocprofv3-test-spm-execute;rocprofv3-test-spm-compare-to-pmc;rocprofv3-test-spm-rocpd-csv-generation"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# SPM derived metrics (mixed basic + derived)
set(DERIVED_COUNTER_LIST TA_FLAT_READ_WAVEFRONTS_sum SQ_WAVES)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-derived-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled=on --spm
        ${DERIVED_COUNTER_LIST} --spm-sample-interval-unit sclk_cycles
        --spm-sample-interval 1200 --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm_derived --output-format json rocpd -o
        spm_derived -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# SPM derived-only (test filtering of implicit base counters)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-derived-only-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled=on --spm
        TA_FLAT_READ_WAVEFRONTS_sum --spm-sample-interval-unit sclk_cycles
        --spm-sample-interval 1200 --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm_derived_only --output-format json rocpd -o
        spm_derived_only -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# rocpd CSV generation for derived-only
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-derived-rocpd-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm_derived_only/rocpd_csv -i
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm_derived_only/spm_derived_only_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${rocprofv3-spm-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-spm-derived-rocpd-csv-generation
    FIXTURES_REQUIRED rocprofv3-test-spm-derived-only-execute
    DISABLED ${IS_DISABLED})

# Validate derived metric results
rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-derived-validate
    TEST_PATHS validate.py
    ARGS --spm-derived-json
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm_derived/spm_derived_results.json
         --spm-derived-only-json
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm_derived_only/spm_derived_only_results.json
         --spm-derived-csv
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm_derived_only/rocpd_csv/out_spm_counter_collection_trace.csv
    CONFIG pytest.ini
    COPY conftest.py
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        "rocprofv3-test-spm-derived-execute;rocprofv3-test-spm-derived-only-execute;rocprofv3-test-spm-derived-rocpd-csv-generation"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# Multi-gpu + multiqueue
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-multigpu
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i spm_input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_multi --output-format json rocpd -o multi --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})
