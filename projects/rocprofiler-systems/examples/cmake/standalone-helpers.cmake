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
# Sets up custom compiler for a target using the launch-compiler wrapper
#
# Uses the rocprof-sys-launch-compiler script to redirect compilation
# to the specified compiler (e.g., hipcc) while keeping CMAKE_CXX_COMPILER
# as the nominal compiler. This avoids enable_language(HIP) which can fail
# on build systems like TheRock that wrap compilers (e.g., resource_info.py).
#
# The launch-compiler script is found relative to the examples directory
# (at ../../scripts/rocprof-sys-launch-compiler) or via PATH/ROCM_PATH.
#
# ARGS:
#   COMPILER: Path to the compiler to use (e.g., hipcc)
#   TARGET: Target to apply custom compilation to
#
function(ROCPROFILER_SYSTEMS_CUSTOM_COMPILATION)
    cmake_parse_arguments(ARG "" "COMPILER;TARGET" "" ${ARGN})

    if(NOT ARG_COMPILER OR NOT ARG_TARGET)
        return()
    endif()

    if(NOT TARGET ${ARG_TARGET})
        return()
    endif()

    if(NOT DEFINED ROCPROFSYS_COMPILE_LAUNCHER)
        find_program(
            ROCPROFSYS_COMPILE_LAUNCHER
            NAMES rocprof-sys-launch-compiler
            HINTS
                ${ROCPROFSYS_EXAMPLE_ROOT_DIR}/../scripts
                ${ROCPROFSYS_EXAMPLE_ROOT_DIR}/../../scripts
                ${ROCM_PATH}
                $ENV{ROCM_PATH}
                /opt/rocm
            PATH_SUFFIXES bin scripts
        )
    endif()

    if(NOT ROCPROFSYS_COMPILE_LAUNCHER)
        message(
            AUTHOR_WARNING
            "rocprof-sys-launch-compiler not found. "
            "Cannot set up custom compilation for ${ARG_TARGET}."
        )
        return()
    endif()

    set(_LAUNCH_CMD
        "${ROCPROFSYS_COMPILE_LAUNCHER} ${ARG_COMPILER} ${CMAKE_CXX_COMPILER}"
    )

    set_property(TARGET ${ARG_TARGET} PROPERTY RULE_LAUNCH_COMPILE "${_LAUNCH_CMD}")
    set_property(TARGET ${ARG_TARGET} PROPERTY RULE_LAUNCH_LINK "${_LAUNCH_CMD}")
endfunction()

# ----------------------------------------------------------------------------
# rocprofiler_systems_checkout_git_submodule()
# Checks out a git submodule or clones a repo if not already present
#
# Mirrors the function from cmake/MacroUtilities.cmake for standalone builds.
#
# ARGS:
#   RELATIVE_PATH: Path relative to WORKING_DIRECTORY
#   WORKING_DIRECTORY: Base directory (default: PROJECT_SOURCE_DIR)
#   TEST_FILE: File to check for existence (default: CMakeLists.txt)
#   REPO_URL: URL to clone if submodule checkout fails
#   REPO_BRANCH: Branch to checkout (default: master)
#   RECURSIVE: Also init recursive submodules
#
function(ROCPROFILER_SYSTEMS_CHECKOUT_GIT_SUBMODULE)
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE;REPO_URL;REPO_BRANCH"
        "ADDITIONAL_CMDS"
        ${ARGN}
    )

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif()

    if(NOT CHECKOUT_REPO_BRANCH)
        set(CHECKOUT_REPO_BRANCH "master")
    endif()

    find_package(Git)
    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")

    if(NOT EXISTS "${_DIR}")
        if(NOT CHECKOUT_REPO_URL)
            message(FATAL_ERROR "submodule directory does not exist")
        endif()
    endif()

    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")
    set(_SUBMODULE "${PROJECT_SOURCE_DIR}/.gitmodules")

    set(_TEST_FILE_EXISTS OFF)
    if(EXISTS "${_TEST_FILE}" AND NOT IS_DIRECTORY "${_TEST_FILE}")
        set(_TEST_FILE_EXISTS ON)
    endif()

    if(_TEST_FILE_EXISTS)
        return()
    endif()

    find_package(Git REQUIRED)

    set(_SUBMODULE_EXISTS OFF)
    if(EXISTS "${_SUBMODULE}" AND NOT IS_DIRECTORY "${_SUBMODULE}")
        set(_SUBMODULE_EXISTS ON)
    endif()

    set(_HAS_REPO_URL OFF)
    if(NOT "${CHECKOUT_REPO_URL}" STREQUAL "")
        set(_HAS_REPO_URL ON)
    endif()

    set(_RECURSE "")
    if(CHECKOUT_RECURSIVE)
        set(_RECURSE "--recursive")
    endif()

    if(NOT _TEST_FILE_EXISTS AND _SUBMODULE_EXISTS)
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET
        )

        if(RET GREATER 0)
            message(STATUS "function(rocprofiler_systems_checkout_git_submodule) failed.")
            message(
                FATAL_ERROR
                "Command: \"${GIT_EXECUTABLE} submodule update --init ${_RECURSE} ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}\""
            )
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT _TEST_FILE_EXISTS AND _HAS_REPO_URL)
        message(
            STATUS
            "Checking out '${CHECKOUT_REPO_URL}' @ '${CHECKOUT_REPO_BRANCH}'..."
        )

        if(EXISTS "${_DIR}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DIR})
        endif()

        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET
        )

        if(CHECKOUT_RECURSIVE AND EXISTS "${_DIR}" AND IS_DIRECTORY "${_DIR}")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                WORKING_DIRECTORY ${_DIR}
                RESULT_VARIABLE RET
            )
        endif()

        if(RET GREATER 0)
            message(STATUS "function(rocprofiler_systems_checkout_git_submodule) failed.")
            message(
                FATAL_ERROR
                "Command: \"${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH} ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}\""
            )
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT EXISTS "${_TEST_FILE}" OR NOT _TEST_FILE_EXISTS)
        message(
            FATAL_ERROR
            "Error checking out submodule: '${CHECKOUT_RELATIVE_PATH}' to '${_DIR}'"
        )
    endif()
endfunction()

# ----------------------------------------------------------------------------
# TheRock build compatibility
#
# When building within TheRock, certain examples must be disabled.
# This mirrors the logic in the main project CMakeLists.txt.
#
option(
    ROCPROFSYS_BUILD_FOR_THEROCK
    "Build rocprofiler-systems examples for use with TheRock"
    OFF
)

if(ROCPROFSYS_BUILD_FOR_THEROCK)
    # Lulesh does not build with TheRock
    if(NOT DEFINED ROCPROFSYS_DISABLE_EXAMPLES)
        set(ROCPROFSYS_DISABLE_EXAMPLES
            "lulesh"
            CACHE STRING
            "Disable building examples"
            FORCE
        )
    else()
        if(NOT "lulesh" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
            list(APPEND ROCPROFSYS_DISABLE_EXAMPLES "lulesh")
        endif()
    endif()
endif()

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
# Setup HIP compilation for standalone GPU example builds
#
# We avoid enable_language(HIP) because it can fail on build systems like
# TheRock that wrap compilers (e.g., resource_info.py). Instead, we find
# hipcc and use it via rocprofiler_systems_custom_compilation() which sets
# RULE_LAUNCH_COMPILE/RULE_LAUNCH_LINK on individual targets.
#
# This mirrors the approach used by the full project build (PR #3519).
#
if(ROCPROFSYS_GFX_TARGETS)
    find_program(
        HIPCC_EXECUTABLE
        NAMES hipcc
        HINTS ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin NO_CACHE
    )
    mark_as_advanced(HIPCC_EXECUTABLE)

    if(HIPCC_EXECUTABLE)
        set(ROCPROFSYS_STANDALONE_HIP_AVAILABLE TRUE CACHE INTERNAL "")
        message(STATUS "[standalone] HIP compilation via hipcc: ${HIPCC_EXECUTABLE}")
    else()
        set(ROCPROFSYS_STANDALONE_HIP_AVAILABLE FALSE CACHE INTERNAL "")
        message(STATUS "[standalone] hipcc not found - GPU examples will be skipped")
    endif()
endif()
