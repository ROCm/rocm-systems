#
# rocprofv3 attachment test for ROCTx pause/resume
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-roctx-pause-resume
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer" "LeakSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning")
else()
    set(LOG_LEVEL "info")
endif()

set(attachment-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

foreach(_MODE normal selected)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-attachment-roctx-pause-resume-${_MODE}
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_roctx_pause_resume_test.sh
            $<TARGET_FILE:roctx-attach-pause-resume>
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${CMAKE_CURRENT_BINARY_DIR}
            ${LOG_LEVEL} out ${_MODE}
        DEPENDS rocprofiler-sdk::rocprofv3 roctx-attach-pause-resume
        TIMEOUT 90
        LABELS "attachment;pause-resume"
        DISABLED ${IS_DISABLED}
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${attachment-env}"
        FAIL_REGULAR_EXPRESSION
            "ERROR|FATAL|Skipping output generation|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "This test is skipped."
        FIXTURES_SETUP rocprofv3-test-attachment-roctx-pause-resume-${_MODE})

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-attachment-roctx-pause-resume-${_MODE}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        TIMEOUT 30
        LABELS "attachment;pause-resume"
        DISABLED ${IS_DISABLED}
        SKIP_REGULAR_EXPRESSION "SKIPPED"
        FIXTURES_REQUIRED rocprofv3-test-attachment-roctx-pause-resume-${_MODE}
        ARGS --json-input
             ${CMAKE_CURRENT_BINARY_DIR}/attachment-roctx-${_MODE}-output/out_results.json
             --skip-if
             ${CMAKE_CURRENT_BINARY_DIR}/attachment-roctx-${_MODE}-output/skipped
             --mode
             ${_MODE})
endforeach()
