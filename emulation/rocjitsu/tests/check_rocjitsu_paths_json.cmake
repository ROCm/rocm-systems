# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_RUNTIME_DIR is required")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}"
        "${TEST_EXECUTABLE}" paths --json
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
)

if(NOT _result EQUAL 0)
    message(
        FATAL_ERROR
        "rocjitsu paths --json failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()

string(
    JSON _runtime_dir
    ERROR_VARIABLE _json_error
    GET "${_stdout}"
    runtime_dir
)
if(_json_error)
    message(
        FATAL_ERROR
        "failed to parse paths JSON: ${_json_error}\n${_stdout}"
    )
endif()
string(JSON _daemon_socket GET "${_stdout}" daemon_socket)
string(JSON _config_handoff_path GET "${_stdout}" config_handoff_path)
string(JSON _config_search_dirs_length LENGTH "${_stdout}" config_search_dirs)

if(NOT _runtime_dir STREQUAL TEST_RUNTIME_DIR)
    message(
        FATAL_ERROR
        "runtime_dir was '${_runtime_dir}', expected '${TEST_RUNTIME_DIR}'"
    )
endif()
if(NOT _daemon_socket STREQUAL "${TEST_RUNTIME_DIR}/daemon.sock")
    message(FATAL_ERROR "daemon_socket was '${_daemon_socket}'")
endif()
if(NOT _config_handoff_path STREQUAL "${TEST_RUNTIME_DIR}/config_path")
    message(FATAL_ERROR "config_handoff_path was '${_config_handoff_path}'")
endif()
if(NOT _config_search_dirs_length GREATER 0)
    message(FATAL_ERROR "config_search_dirs should not be empty")
endif()

if(_stdout MATCHES "default_config_path|invocation_")
    message(FATAL_ERROR "paths JSON exposed an internal field:\n${_stdout}")
endif()
