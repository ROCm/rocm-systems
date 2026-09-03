# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

###############################################################################
# FindHIP.cmake — Compatibility shim for legacy find_package(HIP MODULE) callers.
#
# New projects should use CMake's native HIP language support directly:
#   project(myproject LANGUAGES CXX HIP)
#   find_package(hip CONFIG REQUIRED)
#
# This shim exists so existing callers that do find_package(HIP) or
# find_package(HIP QUIET) continue to work without modification.
# All callers in TheRock only use HIP_FOUND and the hip::host / hip::device
# imported targets, both of which are provided by find_package(hip CONFIG).
###############################################################################

# Delegate to CMake's HIP CONFIG package which provides hip::host, hip::device,
# and hip_VERSION without requiring hipcc or hipconfig.
find_package(hip CONFIG QUIET
    PATHS
        "${HIP_DIR}"
        "${ROCM_PATH}"
        "$ENV{ROCM_PATH}"
        "$ENV{HIP_PATH}"
        /opt/rocm
)

# Populate the legacy HIP_FOUND variable that callers check.
if(hip_FOUND)
    set(HIP_FOUND TRUE)
    set(HIP_VERSION "${hip_VERSION}")
    if(HIP_VERSION)
        string(REPLACE "." ";" _hip_version_list "${HIP_VERSION}")
        list(GET _hip_version_list 0 HIP_VERSION_MAJOR)
        list(GET _hip_version_list 1 HIP_VERSION_MINOR)
        list(GET _hip_version_list 2 HIP_VERSION_PATCH)
    endif()
else()
    set(HIP_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIP
    REQUIRED_VARS HIP_FOUND
    VERSION_VAR   HIP_VERSION
)

###############################################################################
# Legacy macro compatibility stubs
#
# WHY THESE STILL EXIST: The original FindHIP.cmake provided HIP_ADD_LIBRARY,
# HIP_ADD_EXECUTABLE, HIP_COMPILE, and HIP_INCLUDE_DIRECTORIES as macros that
# drove hipcc-based compilation under the hood. Those macros have been removed
# as part of deprecating hipcc/hipconfig, but a number of downstream projects
# (e.g. rocALUTION) still call them after find_package(HIP MODULE). Rather
# than requiring all callers to migrate simultaneously, these stubs let existing
# code continue to compile while we switch the implementation to CMake native
# HIP (add_library / hip::device). Individual libraries can migrate to calling
# add_library directly at their own pace, after which these stubs become dead code.
#
# NOTE: These stubs use no hipcc or hipconfig. They delegate to native CMake
# commands backed by find_package(hip CONFIG), so they are fully compatible
# with the amdclang++ toolchain.
###############################################################################

# Internal helper: parse HIPCC_OPTIONS / CLANG_OPTIONS / NVCC_OPTIONS keyword
# groups out of an argument list. Returns remaining sources/cmake-opts and the
# combined compiler flags from HIPCC_OPTIONS and CLANG_OPTIONS.
macro(_HIP_PARSE_ARGS _out_src _out_cmake_opts _out_cxx_opts)
    set(${_out_src})
    set(${_out_cmake_opts})
    set(${_out_cxx_opts})
    set(_hip_parse_key "")
    foreach(_hip_arg ${ARGN})
        if(_hip_arg STREQUAL "HIPCC_OPTIONS" OR
           _hip_arg STREQUAL "CLANG_OPTIONS" OR
           _hip_arg STREQUAL "NVCC_OPTIONS")
            set(_hip_parse_key "${_hip_arg}")
        elseif(_hip_arg STREQUAL "STATIC"    OR _hip_arg STREQUAL "SHARED"  OR
               _hip_arg STREQUAL "MODULE"    OR _hip_arg STREQUAL "OBJECT"  OR
               _hip_arg STREQUAL "INTERFACE" OR _hip_arg STREQUAL "EXCLUDE_FROM_ALL")
            list(APPEND ${_out_cmake_opts} "${_hip_arg}")
        elseif(_hip_parse_key STREQUAL "HIPCC_OPTIONS" OR
               _hip_parse_key STREQUAL "CLANG_OPTIONS")
            list(APPEND ${_out_cxx_opts} "${_hip_arg}")
        elseif(_hip_parse_key STREQUAL "NVCC_OPTIONS")
            # NVCC options silently ignored; this is an AMD-only build path.
        else()
            list(APPEND ${_out_src} "${_hip_arg}")
        endif()
    endforeach()
    unset(_hip_parse_key)
    unset(_hip_arg)
endmacro()

macro(HIP_ADD_LIBRARY hip_target)
    _HIP_PARSE_ARGS(_hip_src _hip_cmake_opts _hip_cxx_opts ${ARGN})
    add_library(${hip_target} ${_hip_cmake_opts} ${_hip_src})
    if(_hip_cxx_opts)
        target_compile_options(${hip_target} PRIVATE ${_hip_cxx_opts})
    endif()
    if(hip_FOUND AND TARGET hip::device)
        target_link_libraries(${hip_target} PRIVATE hip::device)
    endif()
    unset(_hip_src)
    unset(_hip_cmake_opts)
    unset(_hip_cxx_opts)
endmacro()

macro(HIP_ADD_EXECUTABLE hip_target)
    _HIP_PARSE_ARGS(_hip_src _hip_cmake_opts _hip_cxx_opts ${ARGN})
    add_executable(${hip_target} ${_hip_src})
    if(_hip_cxx_opts)
        target_compile_options(${hip_target} PRIVATE ${_hip_cxx_opts})
    endif()
    if(hip_FOUND AND TARGET hip::device)
        target_link_libraries(${hip_target} PRIVATE hip::device)
    endif()
    unset(_hip_src)
    unset(_hip_cmake_opts)
    unset(_hip_cxx_opts)
endmacro()

# HIP_COMPILE: compile HIP sources into an OBJECT library; caller receives the
# generator expression $<TARGET_OBJECTS:...> for use in a later add_library/executable.
macro(HIP_COMPILE _generated_files)
    _HIP_PARSE_ARGS(_hip_src _hip_cmake_opts _hip_cxx_opts ${ARGN})
    set(_hip_obj_tgt "_hip_compile_${CMAKE_CURRENT_BINARY_DIR}_${CMAKE_CURRENT_LIST_LINE}")
    string(MAKE_C_IDENTIFIER "${_hip_obj_tgt}" _hip_obj_tgt)
    add_library(${_hip_obj_tgt} OBJECT ${_hip_src})
    if(_hip_cxx_opts)
        target_compile_options(${_hip_obj_tgt} PRIVATE ${_hip_cxx_opts})
    endif()
    set(${_generated_files} $<TARGET_OBJECTS:${_hip_obj_tgt}>)
    unset(_hip_src)
    unset(_hip_cmake_opts)
    unset(_hip_cxx_opts)
endmacro()

# HIP_INCLUDE_DIRECTORIES: legacy wrapper around include_directories().
macro(HIP_INCLUDE_DIRECTORIES)
    include_directories(${ARGN})
endmacro()
