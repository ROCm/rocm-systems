# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_PREFIX)
    message(FATAL_ERROR "TEST_PREFIX is required")
endif()
if(NOT DEFINED TEST_CONFIG_SOURCE)
    message(FATAL_ERROR "TEST_CONFIG_SOURCE is required")
endif()
if(NOT DEFINED TEST_CONFIG_RELDIR)
    message(FATAL_ERROR "TEST_CONFIG_RELDIR is required")
endif()

set(_bin_dir "${TEST_PREFIX}/bin")
set(_config_dir "${TEST_PREFIX}/${TEST_CONFIG_RELDIR}")
file(REMOVE_RECURSE "${TEST_PREFIX}")
file(MAKE_DIRECTORY "${_bin_dir}" "${_config_dir}")
file(COPY_FILE "${TEST_EXECUTABLE}" "${_bin_dir}/rocjitsu")
file(
    CHMOD
    "${_bin_dir}/rocjitsu"
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
)
file(READ "${TEST_CONFIG_SOURCE}" _profile_json)
string(
    REPLACE
    "{\"num_threads\": 8, \"cpu_dispatch_threads\": 17, \"async_helper_threads\": 8}"
    "{\"num_threads\": 7, \"cpu_dispatch_threads\": 19, \"async_helper_threads\": 6}"
    _profile_json
    "${_profile_json}"
)
file(WRITE "${_config_dir}/gfx950_mi355x.json" "${_profile_json}")

execute_process(
    COMMAND
        "${_bin_dir}/rocjitsu" --profile mi350x --cpu-thread-budget 31
        --thread-budget-table
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error
)

if(NOT _result EQUAL 0)
    message(
        FATAL_ERROR
        "relocated profile lookup failed with ${_result}\nstdout:\n${_output}\nstderr:\n${_error}"
    )
endif()

if(NOT _output MATCHES "Configured \\| 7 \\| 19 \\| 6 \\| 31")
    message(
        FATAL_ERROR
        "relocated profile lookup used the wrong config:\n${_output}"
    )
endif()
