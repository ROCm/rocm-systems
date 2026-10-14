# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_REGEX)
    message(FATAL_ERROR "TEST_REGEX is required")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}" ${TEST_ARGS}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
)

if(_result EQUAL 0)
    message(
        FATAL_ERROR
        "expected command to fail, but it exited 0\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()

set(_output "${_stdout}${_stderr}")
if(NOT _output MATCHES "${TEST_REGEX}")
    message(
        FATAL_ERROR
        "expected output to match ${TEST_REGEX}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()
