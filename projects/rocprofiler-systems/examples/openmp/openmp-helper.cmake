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
        "Could not find amdclang++. This is required for the OpenMP examples."
    )
endif()

if(NOT COMMAND rocprofiler_systems_custom_compilation)
    rocprofiler_systems_message(
        FATAL_ERROR
        "rocprofiler_systems_custom_compilation() is not available. "
        "The OpenMP examples require the rocprofiler-systems CMake helpers."
    )
endif()
