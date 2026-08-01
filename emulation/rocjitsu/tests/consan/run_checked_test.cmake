# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(DEFINED RJ_EXPECTED_REGEX_FILE)
    file(READ "${RJ_EXPECTED_REGEX_FILE}" RJ_EXPECTED_REGEX)
elseif(NOT DEFINED RJ_EXPECTED_REGEX)
    message(
        FATAL_ERROR
        "RJ_EXPECTED_REGEX or RJ_EXPECTED_REGEX_FILE is required"
    )
endif()

set(_command)
set(_collect_command FALSE)
math(EXPR _last_argument "${CMAKE_ARGC} - 1")
foreach(_index RANGE 0 ${_last_argument})
    set(_argument "${CMAKE_ARGV${_index}}")
    if(_collect_command)
        # Preserve semicolons inside one child argument when building CMake's
        # list-form COMMAND value.
        string(REPLACE ";" "\\;" _argument "${_argument}")
        list(APPEND _command "${_argument}")
    elseif(_argument STREQUAL "--")
        set(_collect_command TRUE)
    endif()
endforeach()

if(NOT _command)
    message(FATAL_ERROR "a test command is required after --")
endif()

execute_process(
    COMMAND ${_command}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
)

if(_stdout)
    message("${_stdout}")
endif()
if(_stderr)
    message("${_stderr}")
endif()

if(NOT "${_result}" STREQUAL "0")
    message(FATAL_ERROR "test command exited with ${_result}")
endif()

set(_combined_output "${_stdout}\n${_stderr}")
if(NOT _combined_output MATCHES "${RJ_EXPECTED_REGEX}")
    message(FATAL_ERROR "test output did not match: ${RJ_EXPECTED_REGEX}")
endif()
