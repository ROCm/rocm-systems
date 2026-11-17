#
# rocprofv3 repeated attachment + ATT test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-attach-twice-att
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

# Attach twice to one live process with identical ATT configuration. Each attachment must
# independently produce raw, metadata, and decoded ATT output.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-attachment-attach-twice-att
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_att_test.sh
        $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} "${attdecoder_LIB_DIR}"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "attachment" "att"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${attachment-env}"
    FAIL_REGULAR_EXPRESSION
        "ERROR|FATAL|Skipping output generation|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    SKIP_REGULAR_EXPRESSION "This test is skipped."
    FIXTURES_SETUP rocprofv3-test-attachment-attach-twice-att)
