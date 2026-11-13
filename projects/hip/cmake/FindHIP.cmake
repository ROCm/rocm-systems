# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

###############################################################################
# FindHIP.cmake: Compatibility shim for legacy find_package(HIP MODULE) callers.
#
# New projects should use CMake's native HIP language support directly:
#   project(myproject LANGUAGES CXX HIP)
#   find_package(hip CONFIG REQUIRED)
#
# This shim exists so existing callers that do find_package(HIP) or
# find_package(HIP QUIET) continue to work without modification.
# Callers use HIP_FOUND, the hip::host / hip::device imported targets, and
# legacy macros (HIP_ADD_LIBRARY, HIP_ADD_EXECUTABLE, HIP_INCLUDE_DIRECTORIES)
# — all provided by this shim without requiring hipcc or hipconfig.
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

# Enable the HIP language so that sources tagged LANGUAGE HIP are compiled by
# CMAKE_HIP_COMPILER (amdclang++) rather than CMAKE_CXX_COMPILER (which may be
# GCC and would choke on -x hip / --offload-arch flags).
if(hip_FOUND)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
    endif()
endif()

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
# HIP_ADD_EXECUTABLE, and HIP_INCLUDE_DIRECTORIES as macros that drove
# hipcc-based compilation under the hood. Those macros have been removed
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

# Compile .cu files and files tagged HIP_SOURCE_PROPERTY_FORMAT as HIP.
# Leave every other source on the host C++ compiler.
macro(_HIP_MARK_HIP_SOURCES)
    foreach(_hip_s ${ARGN})
        get_source_file_property(_hip_header "${_hip_s}" HEADER_FILE_ONLY)
        if(_hip_header)
            continue()
        endif()
        get_source_file_property(_hip_fmt "${_hip_s}" HIP_SOURCE_PROPERTY_FORMAT)
        get_filename_component(_hip_ext "${_hip_s}" LAST_EXT)
        if(_hip_fmt OR _hip_ext STREQUAL ".cu")
            set_source_files_properties("${_hip_s}" PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
    unset(_hip_s)
    unset(_hip_header)
    unset(_hip_fmt)
    unset(_hip_ext)
endmacro()

# Escape a literal > so it does not close the surrounding generator expression.
function(_hip_escape_angle_brackets input output)
    set(_out "")
    set(_depth 0)
    string(LENGTH "${input}" _len)
    set(_i 0)
    while(_i LESS _len)
        string(SUBSTRING "${input}" ${_i} 2 _two)
        if(_two STREQUAL "$<")
            math(EXPR _depth "${_depth} + 1")
            string(APPEND _out "$<")
            math(EXPR _i "${_i} + 2")
            continue()
        endif()
        string(SUBSTRING "${input}" ${_i} 1 _ch)
        if(_ch STREQUAL ">" AND _depth GREATER 0)
            math(EXPR _depth "${_depth} - 1")
            string(APPEND _out ">")
        elseif(_ch STREQUAL ">")
            string(APPEND _out "$<ANGLE-R>")
        else()
            string(APPEND _out "${_ch}")
        endif()
        math(EXPR _i "${_i} + 1")
    endwhile()
    set(${output} "${_out}" PARENT_SCOPE)
endfunction()

# SHELL: removes extra quotes around flags such as "-Wno-#pragma-messages".
# The generator expression applies those flags only to HIP compiles.
macro(_HIP_ADD_HIPCC_OPTIONS hip_target)
    foreach(_hip_o ${ARGN})
        _hip_escape_angle_brackets("${_hip_o}" _hip_o)
        target_compile_options(${hip_target} PRIVATE
            "$<$<COMPILE_LANGUAGE:HIP>:SHELL:${_hip_o}>")
    endforeach()
    unset(_hip_o)
endmacro()

macro(HIP_ADD_LIBRARY hip_target)
    _HIP_PARSE_ARGS(_hip_src _hip_cmake_opts _hip_cxx_opts ${ARGN})
    if(CMAKE_HIP_COMPILER)
        _HIP_MARK_HIP_SOURCES(${_hip_src})
    endif()
    add_library(${hip_target} ${_hip_cmake_opts} ${_hip_src})
    if(_hip_cxx_opts)
        _HIP_ADD_HIPCC_OPTIONS(${hip_target} ${_hip_cxx_opts})
    endif()
    # Forward HIP_HIPCC_FLAGS, such as --offload-arch, to HIP compiles only.
    if(HIP_HIPCC_FLAGS)
        _HIP_ADD_HIPCC_OPTIONS(${hip_target} ${HIP_HIPCC_FLAGS})
    endif()
    if(hip_FOUND AND TARGET hip::host)
        # hip::host provides headers and libamdhip64 without adding -x hip.
        target_link_libraries(${hip_target} PUBLIC hip::host)
    endif()
    unset(_hip_src)
    unset(_hip_cmake_opts)
    unset(_hip_cxx_opts)
endmacro()

macro(HIP_ADD_EXECUTABLE hip_target)
    _HIP_PARSE_ARGS(_hip_src _hip_cmake_opts _hip_cxx_opts ${ARGN})
    if(CMAKE_HIP_COMPILER)
        _HIP_MARK_HIP_SOURCES(${_hip_src})
    endif()
    add_executable(${hip_target} ${_hip_cmake_opts} ${_hip_src})
    if(_hip_cxx_opts)
        _HIP_ADD_HIPCC_OPTIONS(${hip_target} ${_hip_cxx_opts})
    endif()
    if(HIP_HIPCC_FLAGS)
        _HIP_ADD_HIPCC_OPTIONS(${hip_target} ${HIP_HIPCC_FLAGS})
    endif()
    if(hip_FOUND AND TARGET hip::host)
        # PRIVATE for executables. Nothing consumes an executable's link interface.
        target_link_libraries(${hip_target} PRIVATE hip::host)
    endif()
    unset(_hip_src)
    unset(_hip_cmake_opts)
    unset(_hip_cxx_opts)
endmacro()

# HIP_INCLUDE_DIRECTORIES: legacy wrapper around include_directories().
macro(HIP_INCLUDE_DIRECTORIES)
    include_directories(${ARGN})
endmacro()
