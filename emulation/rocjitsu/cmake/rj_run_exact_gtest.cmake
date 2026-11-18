# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# An explicit GoogleTest filter exits successfully when its case is renamed.
# Check the selected case without letting CTest's PASS_REGULAR_EXPRESSION
# override a nonzero test or sanitizer exit status.
foreach(_required IN ITEMS TEST_EXECUTABLE TEST_FILTER)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

get_filename_component(
    _test_executable
    "${TEST_EXECUTABLE}"
    ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_LIST_DIR}"
)
execute_process(
    COMMAND "${_test_executable}" "--gtest_filter=${TEST_FILTER}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _errors
)
if(_output)
    message("${_output}")
endif()
if(_errors)
    message("${_errors}")
endif()
if(NOT "${_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "Filtered test '${TEST_FILTER}' exited with status ${_result}"
    )
endif()
if(NOT _output MATCHES "\\[  PASSED  \\] 1 test\\.")
    message(
        FATAL_ERROR
        "GoogleTest filter '${TEST_FILTER}' did not pass exactly one test"
    )
endif()
