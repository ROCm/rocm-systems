# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RCCL_EXECUTABLE)
    message(FATAL_ERROR "TEST_RCCL_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_RUNTIME_DIR is required")
endif()
if(NOT DEFINED TEST_SHARED_DIR)
    message(FATAL_ERROR "TEST_SHARED_DIR is required")
endif()

file(REMOVE_RECURSE "${TEST_SHARED_DIR}")
file(MAKE_DIRECTORY "${TEST_SHARED_DIR}")

set(_rank_command
    [=[
exec "$RJ_RCCL_EXECUTABLE" --rank="$RANK" --world-size="$WORLD_SIZE" --shared-dir="$RJ_RCCL_SHARED_DIR" --gtest_filter=RcclTest.AllReduce
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}"
        "RJ_RCCL_EXECUTABLE=${TEST_RCCL_EXECUTABLE}"
        "RJ_RCCL_SHARED_DIR=${TEST_SHARED_DIR}" "${TEST_EXECUTABLE}" --profile
        mi350x --num-nodes 1 --nproc-per-node 2 -- /bin/sh -c "${_rank_command}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    TIMEOUT 180
)

if(NOT _result EQUAL 0)
    message(
        FATAL_ERROR
        "process-grid RCCL all-reduce failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()
