# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

foreach(_required IN ITEMS TEST_EXECUTABLE TEST_OUTPUT_FILE)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

# GoogleTest writes XML at the end of the run. Remove the previous result before
# launching it so a crash or timeout cannot leave stale passing results behind.
file(REMOVE "${TEST_OUTPUT_FILE}")
if(EXISTS "${TEST_OUTPUT_FILE}" OR IS_SYMLINK "${TEST_OUTPUT_FILE}")
    message(
        FATAL_ERROR
        "Could not remove previous test results: ${TEST_OUTPUT_FILE}"
    )
endif()

execute_process(
    COMMAND
        "${TEST_EXECUTABLE}" "--gtest_filter=*" --gtest_brief=1
        "--gtest_output=xml:${TEST_OUTPUT_FILE}"
    COMMAND_ERROR_IS_FATAL ANY
)
