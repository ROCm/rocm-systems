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

if(DEFINED RJ_EXPECTED_EXIT)
    set(_expected_result "${RJ_EXPECTED_EXIT}")
else()
    set(_expected_result "0")
endif()
if(NOT "${_result}" STREQUAL "${_expected_result}")
    message(
        FATAL_ERROR
        "test command exited with ${_result}, expected ${_expected_result}"
    )
endif()

set(_combined_output "${_stdout}\n${_stderr}")
if(NOT _combined_output MATCHES "${RJ_EXPECTED_REGEX}")
    message(FATAL_ERROR "test output did not match: ${RJ_EXPECTED_REGEX}")
endif()

if(DEFINED RJ_ADDITIONAL_EXPECTED_REGEX_COUNT)
    set(_additional_regex_count "${RJ_ADDITIONAL_EXPECTED_REGEX_COUNT}")
else()
    set(_additional_regex_count 0)
endif()
if(_additional_regex_count GREATER 0)
    math(EXPR _last_additional_regex "${_additional_regex_count} - 1")
    foreach(_index RANGE 0 ${_last_additional_regex})
        set(_file_variable "RJ_ADDITIONAL_EXPECTED_REGEX_FILE_${_index}")
        if(NOT DEFINED ${_file_variable})
            message(FATAL_ERROR "missing ${_file_variable}")
        endif()
        file(READ "${${_file_variable}}" _additional_regex)
        if(NOT _combined_output MATCHES "${_additional_regex}")
            message(
                FATAL_ERROR
                "test output did not match additional requirement ${_index}: ${_additional_regex}"
            )
        endif()
    endforeach()
endif()
