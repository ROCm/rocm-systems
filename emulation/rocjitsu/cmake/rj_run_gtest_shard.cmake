# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

foreach(_required IN ITEMS TEST_EXECUTABLE TEST_OUTPUT_FILE)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

# GoogleTest writes XML only when the batch finishes. Clear the old report before
# starting the test process.
get_filename_component(TEST_OUTPUT_FILE "${TEST_OUTPUT_FILE}" ABSOLUTE)
get_filename_component(_output_dir "${TEST_OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${_output_dir}")
# EXISTS and rm -f can mistake inaccessible paths for missing files. TOUCH fails
# on access errors and lets non-forced rm handle an initially absent report.
if(NOT EXISTS "${TEST_OUTPUT_FILE}" AND NOT IS_SYMLINK "${TEST_OUTPUT_FILE}")
    file(TOUCH "${TEST_OUTPUT_FILE}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E rm -- "${TEST_OUTPUT_FILE}"
    COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
    COMMAND
        "${TEST_EXECUTABLE}" "--gtest_filter=*" --gtest_brief=1
        "--gtest_output=xml:${TEST_OUTPUT_FILE}"
    COMMAND_ERROR_IS_FATAL ANY
)
