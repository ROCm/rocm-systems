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
    0/0/0/4/127.0.0.1/127.0.0.1/rocjitsu-node-0|\
    1/1/0/4/127.0.0.1/127.0.0.1/rocjitsu-node-0|\
    2/0/1/4/127.0.0.1/127.0.0.1/rocjitsu-node-1|\
    3/1/1/4/127.0.0.1/127.0.0.1/rocjitsu-node-1) ;;
    *) exit 31 ;;
esac
if [ "$MASTER_PORT" != "$ROCJITSU_HEAD_PORT" ]; then
    exit 32
fi
case "$MASTER_PORT" in
    ""|*[!0-9]*) exit 33 ;;
esac
printf "%s\n" "$RANK/$LOCAL_RANK/$ROCJITSU_NODE_RANK/$WORLD_SIZE/$NCCL_HOSTID" > "$RJ_GRID_OUT/$RANK"
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}"
        "${TEST_EXECUTABLE}" --profile mi350x --num-nodes 2 "${TEST_NPROC_FLAG}"
        2 --env "RJ_GRID_OUT=${TEST_OUTPUT_DIR}" --env RANK=caller -- /bin/sh -c
        "${_child_script}"
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

file(GLOB _rank_files LIST_DIRECTORIES false "${TEST_OUTPUT_DIR}/*")
list(LENGTH _rank_files _rank_count)
if(NOT _rank_count EQUAL 4)
    message(
        FATAL_ERROR
        "expected 4 rank files, found ${_rank_count}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()

foreach(_rank RANGE 0 3)
    file(READ "${TEST_OUTPUT_DIR}/${_rank}" _content)
    string(STRIP "${_content}" _content)
    if(_rank EQUAL 0)
        set(_expected "0/0/0/4/rocjitsu-node-0")
    elseif(_rank EQUAL 1)
        set(_expected "1/1/0/4/rocjitsu-node-0")
    elseif(_rank EQUAL 2)
        set(_expected "2/0/1/4/rocjitsu-node-1")
    else()
        set(_expected "3/1/1/4/rocjitsu-node-1")
    endif()
    if(NOT _content STREQUAL _expected)
        message(
            FATAL_ERROR
            "rank ${_rank} wrote ${_content}, expected ${_expected}"
        )
    endif()
endforeach()
