#
# SPM rocpd validation tests
#

set(rocprofv3-spm-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

# SPM rocpd execute (generates both rocpd and json from the same run)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-rocpd-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled=on --spm
        ${COUNTER_LIST} --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1200
        --kernel-trace --hip-trace -d ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format
        rocpd json -o spm -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 60
    LABELS "integration-tests;spm"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# rocpd CSV generation from spm database
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-rocpd-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm/rocpd_csv -i
        ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 60
    LABELS "integration-tests;spm"
    ENVIRONMENT "${rocprofv3-spm-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-spm-rocpd-csv-generation
    FIXTURES_REQUIRED rocprofv3-test-spm-rocpd-execute
    DISABLED ${IS_DISABLED})

# rocpd validation
rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-rocpd-validate
    TEST_PATHS validate.py
    DISCOVERY_ARGS -k "not multigpu"
    ARGS --spm-json
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.json
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.db
         --counter-csv
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/rocpd_csv/out_spm_counter_collection_trace.csv
    CONFIG pytest.ini
    COPY conftest.py
    LABELS "integration-tests;spm"
    FIXTURES_REQUIRED
        "rocprofv3-test-spm-rocpd-execute;rocprofv3-test-spm-rocpd-csv-generation"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# Multi-gpu rocpd execute
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-multigpu-rocpd-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i spm_input.json --hip-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_multi --output-format rocpd -o multi --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 60
    LABELS "integration-tests;spm"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# Multi-gpu rocpd stream_id validation
rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-multigpu-rocpd-validate
    TEST_PATHS validate_multigpu.py
    ARGS --rocpd-input ${CMAKE_CURRENT_BINARY_DIR}/out_multi/multi_results.db
    COPY spm_input.json conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests;spm"
    FIXTURES_REQUIRED "rocprofv3-test-spm-multigpu-rocpd-execute"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})
