if(NOT DEFINED EXPECTED_ROCTX)
    set(EXPECTED_ROCTX 1)
endif()

if(NOT "$ENV{TORCH_PROFILER_EMIT_ROCTX}" STREQUAL "${EXPECTED_ROCTX}")
    message(
        FATAL_ERROR
            "Expected TORCH_PROFILER_EMIT_ROCTX=${EXPECTED_ROCTX}, got '$ENV{TORCH_PROFILER_EMIT_ROCTX}'"
        )
endif()

if(NOT "$ENV{TORCH_PROFILER_ROCTX_SELECTED_REGIONS}" STREQUAL "${EXPECTED_ROCTX}")
    message(
        FATAL_ERROR
            "Expected TORCH_PROFILER_ROCTX_SELECTED_REGIONS=${EXPECTED_ROCTX}, got '$ENV{TORCH_PROFILER_ROCTX_SELECTED_REGIONS}'"
        )
endif()

if(EXPECTED_ROCTX)
    foreach(_var ROCPROF_MARKER_API_TRACE ROCPROF_SELECTED_REGIONS
                 ROCPROF_SELECTED_REGIONS_REF_COUNT)
        if(NOT "$ENV{${_var}}" STREQUAL "1")
            message(FATAL_ERROR "Expected ${_var}=1, got '$ENV{${_var}}'")
        endif()
    endforeach()
endif()
