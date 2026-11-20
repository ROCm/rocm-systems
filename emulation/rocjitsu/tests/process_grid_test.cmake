# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_RUNTIME_DIR is required")
endif()
if(NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_OUTPUT_DIR is required")
endif()
if(NOT DEFINED TEST_NPROC_FLAG)
    set(TEST_NPROC_FLAG "--nproc-per-node")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_DIR}")
file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

set(_child_script
    [=[
case "$RANK/$LOCAL_RANK/$ROCJITSU_NODE_RANK/$WORLD_SIZE/$MASTER_ADDR/$ROCJITSU_HEAD_ADDR/$NCCL_HOSTID" in
    0/0/0/4/127.0.0.1/127.0.0.1/rocjitsu-rank-0) ;;
    1/1/0/4/127.0.0.1/127.0.0.1/rocjitsu-rank-1) ;;
    2/0/1/4/127.0.0.1/127.0.0.1/rocjitsu-rank-2) ;;
    3/1/1/4/127.0.0.1/127.0.0.1/rocjitsu-rank-3) ;;
    *) exit 31 ;;
esac
if [ "$MASTER_PORT" != "$ROCJITSU_HEAD_PORT" ]; then
    exit 32
fi
case "$MASTER_PORT" in
    ""|*[!0-9]*) exit 33 ;;
    0) exit 34 ;;
esac
if [ "$NCCL_P2P_DISABLE/$NCCL_SHM_DISABLE/$NCCL_SOCKET_IFNAME" != "1/1/lo" ]; then
    exit 35
fi
printf "%s\n" "$RANK/$LOCAL_RANK/$ROCJITSU_NODE_RANK/$WORLD_SIZE/$NCCL_HOSTID" > "$RJ_GRID_OUT/rank-$RANK"
printf "%s\n" "$MASTER_PORT" > "$RJ_GRID_OUT/port-$RANK"
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env --unset=NCCL_P2P_DISABLE
        --unset=NCCL_SHM_DISABLE --unset=NCCL_SOCKET_IFNAME
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}" "${TEST_EXECUTABLE}"
        --profile mi350x --num-nodes 2 "${TEST_NPROC_FLAG}" 2 --env
        "RJ_GRID_OUT=${TEST_OUTPUT_DIR}" -- /bin/sh -c "${_child_script}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
)
if(NOT _result EQUAL 0)
    message(
        FATAL_ERROR
        "process-grid launch failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()

file(GLOB _rank_files LIST_DIRECTORIES false "${TEST_OUTPUT_DIR}/rank-*")
list(LENGTH _rank_files _rank_count)
if(NOT _rank_count EQUAL 4)
    message(
        FATAL_ERROR
        "expected 4 rank files, found ${_rank_count}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()

set(_shared_port "")
foreach(_rank RANGE 0 3)
    file(READ "${TEST_OUTPUT_DIR}/rank-${_rank}" _content)
    string(STRIP "${_content}" _content)
    if(_rank EQUAL 0)
        set(_expected "0/0/0/4/rocjitsu-rank-0")
    elseif(_rank EQUAL 1)
        set(_expected "1/1/0/4/rocjitsu-rank-1")
    elseif(_rank EQUAL 2)
        set(_expected "2/0/1/4/rocjitsu-rank-2")
    else()
        set(_expected "3/1/1/4/rocjitsu-rank-3")
    endif()
    if(NOT _content STREQUAL _expected)
        message(
            FATAL_ERROR
            "rank ${_rank} wrote ${_content}, expected ${_expected}"
        )
    endif()

    file(READ "${TEST_OUTPUT_DIR}/port-${_rank}" _port)
    string(STRIP "${_port}" _port)
    if(_port STREQUAL "" OR _port STREQUAL "0")
        message(
            FATAL_ERROR
            "rank ${_rank} wrote invalid rendezvous port '${_port}'"
        )
    endif()
    if(_shared_port STREQUAL "")
        set(_shared_port "${_port}")
    elseif(NOT _port STREQUAL _shared_port)
        message(
            FATAL_ERROR
            "rank ${_rank} wrote rendezvous port ${_port}, expected ${_shared_port}"
        )
    endif()
endforeach()

set(_stdin_script "${TEST_OUTPUT_DIR}/stdin.sh")
file(
    WRITE "${_stdin_script}"
    [=[
#!/bin/sh
printf "grid-input\n" | "$RJ_EXECUTABLE" --profile mi350x --num-nodes 1 --nproc-per-node 1 -- /bin/cat
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/stdin"
        "RJ_EXECUTABLE=${TEST_EXECUTABLE}" /bin/sh "${_stdin_script}"
    RESULT_VARIABLE _stdin_result
    OUTPUT_VARIABLE _stdin_stdout
    ERROR_VARIABLE _stdin_stderr
    TIMEOUT 15
)

if(NOT _stdin_result EQUAL 0 OR NOT _stdin_stdout STREQUAL "grid-input\n")
    message(
        FATAL_ERROR
        "one-rank process-grid stdin failed with ${_stdin_result}\nstdout:\n${_stdin_stdout}\nstderr:\n${_stdin_stderr}"
    )
endif()
