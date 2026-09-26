#
# rocprofv3 interactive attachment detach tests
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-attach-detach
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Disable under sanitizers — attach tests do not play well with them
set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer" "LeakSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

set(attachment-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

# cmake-format: off
# add_attach_detach_test(<mode>)
#
# Registers the execute/validate pair for one interactive detach mechanism. Both
# omit --attach-duration-msec, which is what selects the interactive path:
#   enter  - a newline on rocprofv3's stdin answers the "Press Enter to detach" prompt
#   sigint - SIGINT is delivered to rocprofv3 itself, not to the profiled application
# cmake-format: on
function(add_attach_detach_test MODE)
    set(_name rocprofv3-test-attachment-attach-detach-${MODE})
    set(_outdir ${CMAKE_CURRENT_BINARY_DIR}/attachment-detach-${MODE}-output)

    rocprofiler_add_integration_execute_test(
        ${_name}
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_detach_test.sh
            $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
            ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} out ${MODE}
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 90
        LABELS "attachment"
        DISABLED ${IS_DISABLED}
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${attachment-env}"
        FAIL_REGULAR_EXPRESSION
            "ERROR|FATAL|Skipping output generation|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "This test is skipped."
        FIXTURES_SETUP ${_name})

    rocprofiler_add_integration_validate_test(
        ${_name}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        TIMEOUT 30
        LABELS "attachment"
        DISABLED ${IS_DISABLED}
        SKIP_REGULAR_EXPRESSION "SKIPPED"
        FIXTURES_REQUIRED ${_name}
        ARGS --json-input ${_outdir}/out_results.json --skip-if ${_outdir}/skipped)
endfunction()

add_attach_detach_test(enter)
add_attach_detach_test(sigint)
