if(NOT DEFINED ROCPROFV3)
    message(FATAL_ERROR "ROCPROFV3 is not defined")
endif()

set(_attach_pid 987654321)
set(_attach_config "/tmp/rocprofv3_attach_${_attach_pid}.pkl")
file(REMOVE "${_attach_config}")

execute_process(
    COMMAND "${ROCPROFV3}" --pytorch-trace --attach ${_attach_pid} --echo
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)

if(_result EQUAL 0)
    message(FATAL_ERROR "Expected --pytorch-trace with --attach to fail")
endif()

string(FIND "${_stdout}${_stderr}" "cannot be used with --attach" _error_pos)
if(_error_pos EQUAL -1)
    message(
        FATAL_ERROR
            "Expected the PyTorch attach error, got stdout='${_stdout}' stderr='${_stderr}'"
        )
endif()

if(EXISTS "${_attach_config}")
    file(REMOVE "${_attach_config}")
    message(FATAL_ERROR "Rejected attach left stale configuration: ${_attach_config}")
endif()
