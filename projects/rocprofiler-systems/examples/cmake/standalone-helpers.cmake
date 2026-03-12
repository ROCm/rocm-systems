# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Standalone build helpers for rocprofiler-systems examples
#
# This file provides the CMake functions needed to build examples standalone
# (outside of the full rocprofiler-systems project build).
#
# Usage in example CMakeLists.txt:
#   if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#       include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake)
#   endif()

include_guard(DIRECTORY)

# Set ROCPROFSYS_EXAMPLE_ROOT_DIR if not already set
if(NOT DEFINED ROCPROFSYS_EXAMPLE_ROOT_DIR)
    get_filename_component(
        ROCPROFSYS_EXAMPLE_ROOT_DIR
        "${CMAKE_CURRENT_LIST_DIR}/.."
        ABSOLUTE
    )
endif()

# Include causal-helpers.cmake for causal profiling examples
include(${ROCPROFSYS_EXAMPLE_ROOT_DIR}/causal-helpers.cmake OPTIONAL)

# ----------------------------------------------------------------------------
# rocprofiler_systems_message()
# Wrapper around message() with project prefix
#
macro(ROCPROFILER_SYSTEMS_MESSAGE _TYPE)
    message(${_TYPE} "[rocprofiler-systems] " ${ARGN})
endmacro()

# ----------------------------------------------------------------------------
# check_rocminfo()
# Searches for a given regex in the output of rocminfo
#
# ARGS:
#   _REGEX: The regex to search for
#   _RESULT_VARIABLE: The variable to store the result
#   GET_OUTPUT: If present, return the matching output instead of boolean
#
function(CHECK_ROCMINFO _REGEX _RESULT_VARIABLE)
    cmake_parse_arguments(ARG "GET_OUTPUT" "" "" ${ARGN})

    find_program(
        rocminfo_EXECUTABLE
        NAMES rocminfo
        HINTS ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin
    )

    if(NOT DEFINED ARG_GET_OUTPUT AND _REGEX STREQUAL "")
        message(FATAL_ERROR "Regex is empty, but GET_OUTPUT is not defined")
    endif()

    set(_result FALSE)
    set(_failure FALSE)

    if(rocminfo_EXECUTABLE)
        execute_process(
            COMMAND ${rocminfo_EXECUTABLE}
            RESULT_VARIABLE rocminfo_RET
            OUTPUT_VARIABLE rocminfo_OUTPUT
            ERROR_VARIABLE rocminfo_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(rocminfo_RET EQUAL 0)
            if(NOT _REGEX STREQUAL "")
                string(REGEX MATCHALL "${_REGEX}" rocminfo_OUTPUT "${rocminfo_OUTPUT}")
                if(rocminfo_OUTPUT)
                    set(_result TRUE)
                endif()
            endif()
        else()
            message(AUTHOR_WARNING "rocminfo failed: ${rocminfo_ERROR}")
            set(_failure TRUE)
        endif()
    else()
        message(AUTHOR_WARNING "rocminfo not found")
        set(_failure TRUE)
    endif()

    if(ARG_GET_OUTPUT)
        if(NOT _failure)
            set(${_RESULT_VARIABLE} "${rocminfo_OUTPUT}" PARENT_SCOPE)
        else()
            set(${_RESULT_VARIABLE} "" PARENT_SCOPE)
        endif()
        return()
    endif()

    set(${_RESULT_VARIABLE} ${_result} PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# rocprofiler_systems_get_gfx_archs()
# Auto-detects GPU architectures from the system using rocminfo
#
# ARGS:
#   _VAR: Output variable to store detected architectures (semicolon-separated)
#
function(ROCPROFILER_SYSTEMS_GET_GFX_ARCHS _VAR)
    cmake_parse_arguments(ARG "ECHO" "PREFIX;DELIM;GFX_MATCH" "" ${ARGN})

    if(NOT DEFINED ARG_DELIM)
        set(ARG_DELIM ", ")
    endif()

    if(NOT DEFINED ARG_PREFIX)
        set(ARG_PREFIX "[${PROJECT_NAME}] ")
    endif()

    check_rocminfo("Name:[ \t]+gfx[0-9A-Fa-f][0-9A-Fa-f]+" _RAW_GFXINFO GET_OUTPUT)
    if(NOT _RAW_GFXINFO)
        message(AUTHOR_WARNING "Could not detect GPU architectures")
        set(${_VAR} "" PARENT_SCOPE)
        return()
    endif()

    set(_GFXINFO "")
    foreach(_match IN LISTS _RAW_GFXINFO)
        string(REGEX MATCH "gfx[0-9A-Fa-f]+" _arch "${_match}")
        if(_arch)
            list(APPEND _GFXINFO "${_arch}")
        endif()
    endforeach()

    list(REMOVE_ITEM _GFXINFO "gfx000")
    list(REMOVE_DUPLICATES _GFXINFO)

    if(DEFINED ARG_GFX_MATCH)
        set(_FILTERED_GFXINFO "")
        foreach(_arch IN LISTS _GFXINFO)
            if(_arch MATCHES "${ARG_GFX_MATCH}")
                list(APPEND _FILTERED_GFXINFO "${_arch}")
            endif()
        endforeach()
        set(_GFXINFO "${_FILTERED_GFXINFO}")
    endif()

    if(ARG_ECHO)
        string(REPLACE ";" "${ARG_DELIM}" _GFXINFO_ECHO "${_GFXINFO}")
        message(STATUS "${ARG_PREFIX}System architectures: ${_GFXINFO_ECHO}")
    endif()

    set(${_VAR} "${_GFXINFO}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# rocprofiler_systems_lookup_gfx()
# Classifies AMD GPU architectures into categories (instinct, radeon, apu)
#
# ARGS:
#   _TARGET: The gfx ID to classify (e.g., "gfx90a")
#   _OUTPUT_LIST: Output variable for category list
#
function(ROCPROFILER_SYSTEMS_LOOKUP_GFX _TARGET _OUTPUT_LIST)
    set(INSTINCT_LIST
        "gfx900"
        "gfx906"
        "gfx908"
        "gfx90a"
        "gfx942"
        "gfx950"
    )
    set(RADEON_LIST
        "gfx1012"
        "gfx1011"
        "gfx1010"
        "gfx1032"
        "gfx1031"
        "gfx1030"
        "gfx1102"
        "gfx1101"
        "gfx1100"
        "gfx1200"
        "gfx1201"
        "gfx1202"
    )
    set(APU_LIST
        "gfx1035"
        "gfx1036"
        "gfx1103"
        "gfx1151"
        "gfx1152"
        "gfx1153"
    )

    set(_CATEGORIES "")

    if(_TARGET IN_LIST INSTINCT_LIST)
        list(APPEND _CATEGORIES "instinct")
        check_rocminfo("APU" _is_apu)
        if(_is_apu)
            list(APPEND _CATEGORIES "apu")
        endif()
    endif()
    if(_TARGET IN_LIST RADEON_LIST)
        list(APPEND _CATEGORIES "radeon")
    endif()
    if(_TARGET IN_LIST APU_LIST)
        list(APPEND _CATEGORIES "apu")
    endif()

    if(_CATEGORIES STREQUAL "")
        message(AUTHOR_WARNING "Unknown GFX target: ${_TARGET}. Defaulting to instinct")
        list(APPEND _CATEGORIES "instinct")
    endif()

    set(${_OUTPUT_LIST} "${_CATEGORIES}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# rocprofiler_systems_custom_compilation()
# Sets up custom compiler (hipcc) for a target
#
# In standalone mode, this is a no-op since we handle HIP compilation
# differently (via CMAKE_HIP_COMPILER or enable_language(HIP))
#
function(ROCPROFILER_SYSTEMS_CUSTOM_COMPILATION)
    # No-op in standalone mode - HIP is handled via enable_language(HIP)
    # The full project version uses rocprof-sys-launch-compiler wrapper
endfunction()

# ----------------------------------------------------------------------------
# Setup GPU architecture detection for standalone builds
#
if(NOT DEFINED ROCPROFSYS_GFX_TARGETS OR ROCPROFSYS_GFX_TARGETS STREQUAL "")
    rocprofiler_systems_get_gfx_archs(ROCPROFSYS_GFX_TARGETS)
endif()

set(ROCPROFSYS_GFX_TARGETS
    "${ROCPROFSYS_GFX_TARGETS}"
    CACHE STRING
    "GPU architectures to compile for (semicolon-separated)"
)

if(ROCPROFSYS_GFX_TARGETS)
    message(STATUS "[standalone] Detected GPU targets: ${ROCPROFSYS_GFX_TARGETS}")
else()
    message(STATUS "[standalone] No GPU targets detected")
endif()

# ----------------------------------------------------------------------------
# Enable HIP language for standalone GPU example builds
#
# For standalone builds, we enable HIP as a first-class language instead of
# using the launch-compiler wrapper approach used by the full project.
#
# CMake's HIP language requires the ROCm clang directly (not hipcc wrapper)
#
if(ROCPROFSYS_GFX_TARGETS)
    # Find ROCm's clang compiler (required for CMake HIP language)
    find_program(
        _STANDALONE_AMDCLANG
        NAMES amdclang++ clang++
        HINTS ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin llvm/bin
    )

    if(_STANDALONE_AMDCLANG)
        # Set HIP compiler to clang (CMake doesn't accept hipcc wrapper)
        set(CMAKE_HIP_COMPILER ${_STANDALONE_AMDCLANG} CACHE FILEPATH "HIP compiler")

        # Set architectures for HIP targets
        set(CMAKE_HIP_ARCHITECTURES
            ${ROCPROFSYS_GFX_TARGETS}
            CACHE STRING
            "HIP architectures"
        )

        # Enable HIP language
        include(CheckLanguage)
        check_language(HIP)
        if(CMAKE_HIP_COMPILER)
            enable_language(HIP)
            message(
                STATUS
                "[standalone] HIP language enabled with compiler: ${CMAKE_HIP_COMPILER}"
            )
        endif()
    else()
        message(STATUS "[standalone] amdclang++ not found - HIP language not enabled")
    endif()

    unset(_STANDALONE_AMDCLANG)
endif()
