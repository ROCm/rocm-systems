if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_RUNTIME_DIR is required")
endif()

find_program(_unshare_EXECUTABLE unshare)
if(NOT _unshare_EXECUTABLE)
    message(
        STATUS
        "unshare not found; skipping PID namespace process-grid check"
    )
    return()
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}"
        "${_unshare_EXECUTABLE}" --fork --pid "${TEST_EXECUTABLE}" --profile
        mi350x --num-nodes 1 --nproc-per-node 1 -- /bin/true
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    TIMEOUT 15
)

if(_result EQUAL 0)
    return()
endif()

if(
    _stderr
        MATCHES
        "unshare:.*(Operation not permitted|invalid option|unrecognized option|failed)"
)
    message(
        STATUS
        "unshare cannot create a PID namespace here; skipping PID namespace process-grid check"
    )
    return()
endif()

message(
    FATAL_ERROR
    "process-grid PID namespace check failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
)
