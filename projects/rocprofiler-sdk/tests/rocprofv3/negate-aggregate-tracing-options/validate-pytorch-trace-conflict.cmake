if(NOT DEFINED ROCPROFV3)
    message(FATAL_ERROR "ROCPROFV3 is required")
endif()

if(NOT DEFINED CONFLICT)
    message(FATAL_ERROR "CONFLICT is required")
elseif(CONFLICT STREQUAL "collection-period")
    set(_conflicting_args --collection-period 0:1:1)
    set(_expected_error
        "--selected-regions and --collection-period are mutually exclusive")
elseif(CONFLICT STREQUAL "att-consecutive-kernels")
    set(_conflicting_args --att-consecutive-kernels 1)
    set(_expected_error
        "--selected-regions and --att-consecutive-kernels are mutually exclusive")
elseif(CONFLICT STREQUAL "att-no-intercept")
    set(_conflicting_args --att-no-intercept)
    set(_expected_error "--pytorch-trace cannot be used with --att-no-intercept")
else()
    message(FATAL_ERROR "Unknown CONFLICT value: ${CONFLICT}")
endif()

execute_process(
    COMMAND "${ROCPROFV3}" --pytorch-trace ${_conflicting_args} -- "${CMAKE_COMMAND}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)

if(_result EQUAL 0)
    message(FATAL_ERROR "Expected --pytorch-trace conflict '${CONFLICT}' to fail")
endif()

set(_output "${_stdout}\n${_stderr}")
string(FIND "${_output}" "${_expected_error}" _error_position)
if(_error_position EQUAL -1)
    message(
        FATAL_ERROR
            "Expected error '${_expected_error}' for conflict '${CONFLICT}', got:\n${_output}"
        )
endif()
