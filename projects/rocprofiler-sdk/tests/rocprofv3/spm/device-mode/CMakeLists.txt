#
# SPM device-mode rocpd validation tests
#

# SPM device-mode execute
rocprofiler_add_integration_execute_test(
    rocprofv3-test-spm-device-mode-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --spm-beta-enabled=on --spm
        SQ_WAVES --spm-sample-interval-unit sclk_cycles --spm-sample-interval
        8000 --spm-gpu-index 0 --spm-device-mode -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_device_spm --output-format rocpd -o device_spm
        -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 100
    LABELS "integration-tests;spm"
    ENVIRONMENT "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})

# SPM device-mode rocpd validation
rocprofiler_add_integration_validate_test(
    rocprofv3-test-spm-device-mode-validate
    TEST_PATHS validate.py
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/out_device_spm/device_spm_results.db
    CONFIG pytest.ini
    COPY conftest.py
    LABELS "integration-tests;spm"
    FIXTURES_REQUIRED "rocprofv3-test-spm-device-mode-execute"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED ${IS_DISABLED})
