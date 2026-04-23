# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# thread-limit tests
#
# -------------------------------------------------------------------------------------- #

if(NOT TARGET thread-limit)
    return()
endif()

set(_thread_limit_environment
    "${_base_environment}"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_COUT_OUTPUT=ON"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_SAMPLING_FREQ=250"
    "ROCPROFSYS_VERBOSE=2"
    "ROCPROFSYS_LOG_LEVEL=trace"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,peak_rss,page_rss"
)

math(EXPR THREAD_VAL_1 "${ROCPROFSYS_MAX_THREADS} - 1")
math(EXPR THREAD_VAL_2 "${ROCPROFSYS_MAX_THREADS} + 24")

set(THREAD_VALUES ${THREAD_VAL_1} ${THREAD_VAL_2} ${ROCPROFSYS_MAX_THREADS})

# Loop over thread values
foreach(THREADS IN LISTS THREAD_VALUES)
    set(THREAD_PASS_VALUE ${THREADS})
    math(EXPR THREAD_FAIL_VALUE "${THREADS} + 1")
    if(${THREADS} GREATER_EQUAL ${ROCPROFSYS_MAX_THREADS})
        math(EXPR THREAD_PASS_VALUE "${ROCPROFSYS_MAX_THREADS} - 1")
        math(EXPR THREAD_FAIL_VALUE "${ROCPROFSYS_MAX_THREADS} + 1")
    endif()

    set(_thread_limit_pass_regex "\\|${THREAD_PASS_VALUE}>>>")
    set(_thread_limit_fail_regex "\\|${THREAD_FAIL_VALUE}>>>|ROCPROFSYS_ABORT_FAIL_REGEX")

    # Unique test name
    set(_test_name thread-limit-${THREADS})

    # Add test
    rocprofiler_systems_add_test(
        SKIP_BASELINE
        NAME ${_test_name}
        TARGET thread-limit
        LABELS "max-threads"
        REWRITE_ARGS -e -v 2 -i 1024 --label return args
        RUNTIME_ARGS -e -v 1 -i 1024 --label return args
        RUN_ARGS 35 2 ${THREADS}
        SAMPLING_TIMEOUT 480
        REWRITE_TIMEOUT 480
        RUNTIME_TIMEOUT 480
        RUNTIME_PASS_REGEX "${_thread_limit_pass_regex}"
        SAMPLING_PASS_REGEX "${_thread_limit_pass_regex}"
        REWRITE_RUN_PASS_REGEX "${_thread_limit_pass_regex}"
        RUNTIME_FAIL_REGEX "${_thread_limit_fail_regex}"
        SAMPLING_FAIL_REGEX "${_thread_limit_fail_regex}"
        REWRITE_RUN_FAIL_REGEX "${_thread_limit_fail_regex}"
        ENVIRONMENT "${_thread_limit_environment}"
    )
endforeach()
