# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

include_guard(DIRECTORY)

find_program(
    amdclangpp_EXECUTABLE
    NAMES amdclang++
    HINTS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin llvm/bin
)
mark_as_advanced(amdclangpp_EXECUTABLE)

if(NOT amdclangpp_EXECUTABLE)
    rocprofiler_systems_message(
        FATAL_ERROR
        "Could not find amdclang++. This is required for the OpenMP examples. "
        "Append openmp to ROCPROFSYS_DISABLE_EXAMPLES to prevent these examples from being built."
    )
endif()

if(NOT COMMAND rocprofiler_systems_custom_compilation)
    rocprofiler_systems_message(
        FATAL_ERROR
        "rocprofiler_systems_custom_compilation() is not available. "
        "The OpenMP examples require the rocprofiler-systems CMake helpers. "
        "Append openmp to ROCPROFSYS_DISABLE_EXAMPLES to prevent these examples from being built."
    )
endif()

# TODO: Move this to a top level helper in examples and reuse across all examples
# Outputs a bool if the _EXAMPLE_NAME is in ROCPROFSYS_DISABLE_EXAMPLES
function(CHECK_ROCPROFSYS_DISABLE_EXAMPLES _EXAMPLE_NAME _OUTPUT)
    if(ROCPROFSYS_DISABLE_EXAMPLES)
        if("${_EXAMPLE_NAME}" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
            set(${_OUTPUT} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()
    set(${_OUTPUT} FALSE PARENT_SCOPE)
endfunction()
