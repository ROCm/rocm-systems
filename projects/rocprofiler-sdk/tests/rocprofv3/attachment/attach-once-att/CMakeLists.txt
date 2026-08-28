#
# rocprofv3 attachment + ATT test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-attach-once-att
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

find_library(
    attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${ROCM_PATH}
    PATH_SUFFIXES lib)

cmake_path(GET attdecoder_LIBRARY PARENT_PATH attdecoder_LIB_DIR)

set(attachment-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

# Execute phase: launch attachment-test, attach rocprofv3 with ATT enabled for 2 seconds,
# then stop the app.  The ATT library directory is passed as the sixth positional argument
# to the shell script.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-attachment-attach-once-att
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_att_test.sh
        $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} out "${attdecoder_LIB_DIR}"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 90
    LABELS "attachment" "att"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${attachment-env}"
    FAIL_REGULAR_EXPRESSION
        "ERROR|FATAL|Skipping output generation|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    SKIP_REGULAR_EXPRESSION "This test is skipped."
    FIXTURES_SETUP rocprofv3-test-attachment-attach-once-att)

# Validate phase: check the standard profiling output (agent info, kernel dispatch, memory
# copy, HSA API) and the ATT-specific output (filenames manifest, wave files).
rocprofiler_add_integration_validate_test(
    rocprofv3-test-attachment-attach-once-att
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 30
    LABELS "attachment" "att"
    DISABLED ${IS_DISABLED}
    DISABLED_CODECOV
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-attachment-attach-once-att
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/attachment-att-output/out_results.json
         --att-output-dir ${CMAKE_CURRENT_BINARY_DIR}/attachment-att-output --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-att-output/skipped)
