# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

include_guard(DIRECTORY)

# Compilers

# AMD LLVM compilers
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

find_program(
    amdflang_EXECUTABLE
    NAMES amdflang
    HINTS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin llvm/bin
)
mark_as_advanced(amdflang_EXECUTABLE)

# HIP Compiler
find_program(
    HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    NO_CACHE
)
mark_as_advanced(HIPCC_EXECUTABLE)

# Check the compiler version (works for amdclang++, amdflang, hipcc, etc.)
# For HIPCC, this returns the underlying CLANG version, not HIP version
function(CHECK_COMPILER_VERSION _COMPILER _OUTPUT)
    execute_process(
        COMMAND ${_COMPILER} --version
        OUTPUT_VARIABLE _VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    # Match "clang version X" or "flang-new version X" (AMD compiler line)
    # This skips "HIP version:" which appears first in hipcc output
    string(REGEX MATCH "(clang|flang-new) version ([0-9]+)" _MATCH "${_VERSION_OUTPUT}")
    set(${_OUTPUT} "${CMAKE_MATCH_2}" PARENT_SCOPE)
endfunction()

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

# Handle example installation
function(ROCPROFILER_SYSTEMS_INSTALL_HPC_EXAMPLE _EXAMPLE_NAME)
    if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET ${_EXAMPLE_NAME})
        install(
            TARGETS ${_EXAMPLE_NAME}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            COMPONENT rocprofiler-systems-examples
        )
    endif()
endfunction()
