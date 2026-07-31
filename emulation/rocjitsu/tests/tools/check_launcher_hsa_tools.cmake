# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT "$ENV{HSA_TOOLS_DISABLE_REGISTER}" STREQUAL "1")
    message(
        FATAL_ERROR
        "launcher did not select the HSA_TOOLS_LIB callback path"
    )
endif()

set(expected_prefix "$ENV{RJ_EXPECTED_HSA_TOOLS_PREFIX}")
set(actual_tools "$ENV{HSA_TOOLS_LIB}")
string(REPLACE " " ";" actual_tools_list "${actual_tools}")
list(LENGTH actual_tools_list actual_tool_count)
if(actual_tool_count LESS 1)
    message(
        FATAL_ERROR
        "HSA_TOOLS_LIB does not contain the appended rocJITsu hook"
    )
endif()
list(POP_BACK actual_tools_list actual_appended_tool)
list(JOIN actual_tools_list " " actual_prefix)
if(NOT actual_prefix STREQUAL expected_prefix)
    message(
        FATAL_ERROR
        "unexpected HSA_TOOLS_LIB prefix/order: '${actual_tools}'"
    )
endif()

# The launcher preserves existing tools and their order literally. Only its
# appended hook is compared by canonical identity because launcher discovery
# may reach the same build artifact through a symlinked build-tree spelling.
file(REAL_PATH "${actual_appended_tool}" actual_appended_tool_real)
file(
    REAL_PATH "$ENV{RJ_EXPECTED_APPENDED_HSA_TOOL}"
    expected_appended_tool_real
)
if(NOT actual_appended_tool_real STREQUAL expected_appended_tool_real)
    message(
        FATAL_ERROR
        "unexpected appended HSA tool: '${actual_appended_tool}' resolves to "
        "'${actual_appended_tool_real}', expected '${expected_appended_tool_real}'"
    )
endif()

message(STATUS "rocjitsu launcher preserved and ordered HSA tools")
