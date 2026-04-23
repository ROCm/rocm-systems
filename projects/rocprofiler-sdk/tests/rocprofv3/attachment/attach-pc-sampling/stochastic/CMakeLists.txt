#
# rocprofv3 attachment test -- PC sampling (stochastic) with attach/detach
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-attach-pc-sampling-stochastic
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_sdk_pc_sampling_stochastic_disabled(IS_PC_SAMPLING_STOCHASTIC_DISABLED)

set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer")

if(IS_PC_SAMPLING_STOCHASTIC_DISABLED
   OR (ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES))
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

# PC sampling emits a high volume of INFO-level buffer traces; 'warning' keeps -V output
# readable (also required under LeakSanitizer).
set(LOG_LEVEL "warning")

set(PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

set(attachment-pc-sampling-stochastic-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set(ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX
    "PC sampling unavailable|PC sampling configuration is not supported")

# Execute: launch exec-mask-manipulation in background, attach with PC sampling
# (stochastic)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-attachment-attach-pc-sampling-stochastic
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/../run_attach_pc_sampling_test.sh
        $<TARGET_FILE:exec-mask-manipulation> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} out stochastic cycles 1048576 5000
        1048576
    DEPENDS rocprofiler-sdk::rocprofv3 exec-mask-manipulation
    TIMEOUT 90
    LABELS "integration-tests;attachment;pc-sampling;stochastic"
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${attachment-pc-sampling-stochastic-env}"
    SKIP_REGULAR_EXPRESSION
        "This test is skipped.|${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-attachment-attach-pc-sampling-stochastic
    FAIL_REGULAR_EXPRESSION "FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Validate: pytest checks on the collected PC sampling CSVs
rocprofiler_add_integration_validate_test(
    rocprofv3-test-attachment-attach-pc-sampling-stochastic-validate
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 30
    LABELS "integration-tests;attachment;pc-sampling;stochastic"
    DISABLED ${IS_DISABLED}
    SKIP_REGULAR_EXPRESSION "SKIPPED|${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-attachment-attach-pc-sampling-stochastic
    ARGS --input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-pc-sampling-output/out_pc_sampling_stochastic.csv
         --all-sampled
         False
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-pc-sampling-output/skipped)
