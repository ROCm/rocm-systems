# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_PARENT_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_PARENT_RUNTIME_DIR is required")
endif()

execute_process(
    COMMAND id -u
    OUTPUT_VARIABLE _uid
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _id_result
)
if(NOT _id_result EQUAL 0)
    message(FATAL_ERROR "id -u failed")
endif()

set(_default_runtime_dir "/tmp/rocjitsu-${_uid}")
set(_default_socket "${_default_runtime_dir}/daemon.sock")
if(EXISTS "${_default_socket}")
    message(
        "ROCJITSU_EMPTY_RUNTIME_OVERRIDE_SKIP: ${_default_socket} already exists"
    )
    return()
endif()

file(MAKE_DIRECTORY "${TEST_PARENT_RUNTIME_DIR}" "${_default_runtime_dir}")
file(WRITE "${TEST_PARENT_RUNTIME_DIR}/daemon.sock" "")
file(WRITE "${_default_socket}" "")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_PARENT_RUNTIME_DIR}" "XDG_RUNTIME_DIR="
        "RJ_EXPECTED_INVOCATION_DIR=${_default_runtime_dir}"
        "${TEST_EXECUTABLE}" --attach --env "ROCJITSU_RUNTIME_DIR=" -- /bin/sh
        -c [=[test "$ROCJITSU_INVOCATION_DIR" = "$RJ_EXPECTED_INVOCATION_DIR"]=]
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
)

file(REMOVE "${_default_socket}" "${TEST_PARENT_RUNTIME_DIR}/daemon.sock")

if(NOT _result EQUAL 0)
    message(
        FATAL_ERROR
        "empty runtime override failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()
