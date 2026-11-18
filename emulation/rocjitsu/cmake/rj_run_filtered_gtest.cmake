# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Validate one exact GoogleTest filter before running it through rocjitsu.
# GoogleTest normally exits successfully when a filter selects no tests, which
# can make renamed hardware tests appear to pass without executing anything.

foreach(_required IN ITEMS TEST_EXECUTABLE TEST_FILTER)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

execute_process(
    COMMAND
        "${TEST_EXECUTABLE}" --gtest_list_tests "--gtest_filter=${TEST_FILTER}"
    RESULT_VARIABLE _list_result
    OUTPUT_VARIABLE _listed_tests
    ERROR_VARIABLE _list_errors
)
if(NOT _list_result EQUAL 0)
    message(
        FATAL_ERROR
        "Failed to list tests from ${TEST_EXECUTABLE}: ${_list_errors}"
    )
endif()

set(_selected FALSE)
string(REPLACE "\n" ";" _test_lines "${_listed_tests}")
foreach(_line IN LISTS _test_lines)
    if(_line MATCHES "^  [^# \t]+")
        set(_selected TRUE)
        break()
    endif()
endforeach()
if(NOT _selected)
    message(
        FATAL_ERROR
        "GoogleTest filter '${TEST_FILTER}' selects no tests in ${TEST_EXECUTABLE}"
    )
endif()

if(LIST_ONLY)
    return()
endif()

foreach(_required IN ITEMS TEST_LAUNCHER TEST_CONFIG)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required when LIST_ONLY is false")
    endif()
endforeach()

execute_process(
    COMMAND
        "${TEST_LAUNCHER}" --config "${TEST_CONFIG}" -- "${TEST_EXECUTABLE}"
        "--gtest_filter=${TEST_FILTER}"
    RESULT_VARIABLE _run_result
    OUTPUT_VARIABLE _run_output
    ERROR_VARIABLE _run_errors
)
if(_run_output)
    message("${_run_output}")
endif()
if(_run_errors)
    message("${_run_errors}")
endif()
if(NOT _run_result EQUAL 0)
    message(
        FATAL_ERROR
        "Filtered test '${TEST_FILTER}' exited with status ${_run_result}"
    )
endif()

# GoogleTest exits 0 when a test calls GTEST_SKIP(). That is the right outcome
# for a registration whose prerequisite may genuinely be absent, but not for one
# that runs against a simulated agent the launcher is configured to supply: there
# a skip means the config, interposer, or launcher regressed. FAIL_ON_SKIP turns
# that into a hard failure so it cannot report green.
if(FAIL_ON_SKIP AND _run_output MATCHES "\\[  SKIPPED \\]")
    message(
        FATAL_ERROR
        "Filtered test '${TEST_FILTER}' was SKIPPED, but this registration "
        "requires it to run"
    )
endif()
