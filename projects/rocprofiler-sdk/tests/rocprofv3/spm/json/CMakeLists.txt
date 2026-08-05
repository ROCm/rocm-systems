#
# SPM JSON validation tests
#

# SPM JSON execute
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled=on --spm
        ${COUNTER_LIST} --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1200
        --kernel-trace --hip-trace -d ${CMAKE_CURRENT_BINARY_DIR}/out_spm --output-format
        json -o spm -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# PMC compare execute
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-json-compare-to-pmc
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

# JSON validation
rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-json-validate
    TEST_PATHS validate.py
    DISCOVERY_ARGS -k "not multigpu"
    ARGS --pmc-json ${CMAKE_CURRENT_BINARY_DIR}/out_spm/pmc_results.json --spm-json
         ${CMAKE_CURRENT_BINARY_DIR}/out_spm/spm_results.json
    CONFIG pytest.ini
    COPY spm_input.json conftest.py
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        "rocprofv3-test-spm-json-execute;rocprofv3-test-spm-json-compare-to-pmc"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# Multi-gpu JSON execute
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-multigpu-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i spm_input.json --hip-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_multi --output-format json -o multi --
        $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 60
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# Multi-gpu JSON validation
rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-multigpu-json-validate
    TEST_PATHS validate.py
    DISCOVERY_ARGS -k multigpu
    ARGS --spm-json ${CMAKE_CURRENT_BINARY_DIR}/out_multi/multi_results.json
    CONFIG pytest.ini
    COPY conftest.py
    LABELS "integration-tests"
    FIXTURES_REQUIRED "rocprofv3-test-spm-multigpu-json-execute"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})
