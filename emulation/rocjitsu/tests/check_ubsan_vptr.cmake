# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

execute_process(
    COMMAND "${PROBE}"
    RESULT_VARIABLE valid_result
    OUTPUT_VARIABLE valid_stdout
    ERROR_VARIABLE valid_stderr
)
if(NOT valid_result STREQUAL "0" OR valid_stderr MATCHES "runtime error:")
    message(
        FATAL_ERROR
        "Valid downcast failed (${valid_result}): ${valid_stdout}${valid_stderr}"
    )
endif()

execute_process(
    COMMAND "${PROBE}" invalid
    RESULT_VARIABLE invalid_result
    OUTPUT_VARIABLE invalid_stdout
    ERROR_VARIABLE invalid_stderr
)
if(EXPECT_VPTR)
    if(
        invalid_result STREQUAL "0"
        OR NOT invalid_stderr MATCHES "runtime error: downcast of address"
    )
        message(
            FATAL_ERROR
            "Expected a failing vptr diagnostic (${invalid_result}): ${invalid_stdout}${invalid_stderr}"
        )
    endif()
elseif(
    NOT invalid_result STREQUAL "0"
    OR invalid_stderr MATCHES "runtime error:"
)
    message(
        FATAL_ERROR
        "vptr-disabled probe failed (${invalid_result}): ${invalid_stdout}${invalid_stderr}"
    )
endif()
