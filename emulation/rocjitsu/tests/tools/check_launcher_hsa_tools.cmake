# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT "$ENV{HSA_TOOLS_DISABLE_REGISTER}" STREQUAL "1")
    message(
        FATAL_ERROR
        "launcher did not select the HSA_TOOLS_LIB callback path"
    )
endif()

set(expected_tools
    "$ENV{RJ_EXPECTED_HSA_TOOLS_PREFIX} $ENV{RJ_EXPECTED_APPENDED_HSA_TOOL}"
)
if(NOT "$ENV{HSA_TOOLS_LIB}" STREQUAL "${expected_tools}")
    message(
        FATAL_ERROR
        "unexpected HSA_TOOLS_LIB: '$ENV{HSA_TOOLS_LIB}', expected '${expected_tools}'"
    )
endif()

message(STATUS "rocjitsu launcher preserved and ordered HSA tools")
