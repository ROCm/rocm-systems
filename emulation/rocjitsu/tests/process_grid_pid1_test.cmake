# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_RUNTIME_DIR is required")
endif()

find_program(_unshare_EXECUTABLE unshare)
if(NOT _unshare_EXECUTABLE)
    message("ROCJITSU_PROCESS_GRID_PID1_SKIP: unshare not found")
    return()
endif()

set(_rocjitsu_command
    "${TEST_EXECUTABLE}"
    --profile
    mi350x
    --num-nodes
    1
    --nproc-per-node
    1
    --
    /bin/true
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/pid"
        "${_unshare_EXECUTABLE}" --fork --pid ${_rocjitsu_command}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    TIMEOUT 15
)

if(_result EQUAL 0)
    return()
endif()

set(_first_error "${_stderr}")
if(
    _stderr
        MATCHES
        "unshare:.*(Operation not permitted|not permitted|permission denied)"
)
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" -E env
            "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/user-pid"
            "${_unshare_EXECUTABLE}" --user --map-root-user --fork --pid
            ${_rocjitsu_command}
        RESULT_VARIABLE _user_result
        OUTPUT_VARIABLE _user_stdout
        ERROR_VARIABLE _user_stderr
        TIMEOUT 15
    )
    if(_user_result EQUAL 0)
        return()
    endif()
    set(_result "${_user_result}")
    string(APPEND _stdout "\nuser namespace stdout:\n${_user_stdout}")
    string(
        APPEND _stderr
        "\nfirst stderr:\n${_first_error}\nuser namespace stderr:\n${_user_stderr}"
    )
endif()

if(
    _stderr
        MATCHES
        "unshare:.*(Operation not permitted|not permitted|permission denied|invalid option|unrecognized option)"
)
    message("ROCJITSU_PROCESS_GRID_PID1_SKIP: namespace support unavailable")
    return()
endif()

message(
    FATAL_ERROR
    "process-grid PID namespace check failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
)
