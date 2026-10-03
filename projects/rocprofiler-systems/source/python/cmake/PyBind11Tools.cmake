# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#
function(ROCPROFILER_SYSTEMS_FIND_PYTHON _VAR)
    set(options REQUIRED QUIET)
    set(args VERSION ROOT_DIR)
    set(kwargs COMPONENTS)
    cmake_parse_arguments(ARG "${options}" "${args}" "${kwargs}" ${ARGN})

    if(ARG_QUIET)
        set(_QUIET "QUIET")
    endif()

    if(ARG_VERSION)
        set(_EXACT "EXACT")
    endif()

    if(ARG_REQUIRED)
        set(_FIND_REQUIREMENT "REQUIRED")
    endif()

    if(NOT ARG_COMPONENTS)
        set(ARG_COMPONENTS Interpreter Development)
    endif()

    rocprofiler_systems_reset_python3_cache()

    set(Python3_ROOT_DIR "${ARG_ROOT_DIR}")
    set(Python3_FIND_STRATEGY "LOCATION")
    set(Python3_FIND_VIRTUALENV "FIRST")
    set(Python3_ARTIFACTS_INTERACTIVE OFF)

    find_package(
        Python3
        ${ARG_VERSION}
        ${_EXACT}
        ${_QUIET}
        MODULE
        ${_FIND_REQUIREMENT}
        COMPONENTS ${ARG_COMPONENTS}
    )

    set(${_VAR}_FOUND "${Python3_FOUND}" PARENT_SCOPE)
    if(NOT Python3_FOUND)
        set(${_VAR}_EXECUTABLE "" PARENT_SCOPE)
        set(${_VAR}_ROOT_DIR "" PARENT_SCOPE)
        set(${_VAR}_VERSION "" PARENT_SCOPE)
        return()
    endif()

    set(${_VAR}_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
    set(${_VAR}_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}" PARENT_SCOPE)

    # Derive the interpreter's root dir (sys.prefix) from its path rather than spawning a
    # Python subprocess: the directory containing bin/<executable> is sys.prefix for every
    # layout this project builds against (system install, conda env, venv). TheRock's
    # superbuild relies on this same derivation when forwarding Python root dirs to this
    # project, so this is a proven equivalence, not a new assumption.
    cmake_path(GET Python3_EXECUTABLE PARENT_PATH _bin_dir)
    cmake_path(GET _bin_dir PARENT_PATH _root_dir)
    set(${_VAR}_ROOT_DIR "${_root_dir}" PARENT_SCOPE)
endfunction()
#
# Internal: unset cached Python3 discovery variables so a subsequent find_package(Python3)
# call for a different version/root dir is not short-circuited by a previous result.
macro(ROCPROFILER_SYSTEMS_RESET_PYTHON3_CACHE)
    foreach(
        _VAR
        _Python3_Compiler_REASON_FAILURE
        _Python3_Interpreter_REASON_FAILURE
        _Python3_Development_LIBRARY_REASON_FAILURE
        _Python3_Development_SABI_LIBRARY_REASON_FAILURE
        _Python3_DEVELOPMENT_MODULE_SIGNATURE
        _Python3_EXECUTABLE
        _Python3_INCLUDE_DIR
        _Python3_INTERPRETER_PROPERTIES
        _Python3_INTERPRETER_SIGNATURE
        _Python3_LIBRARY_RELEASE
        Python3_EXECUTABLE
        Python3_INCLUDE_DIR
        Python3_INTERPRETER_ID
        Python3_STDLIB
        Python3_STDARCH
        Python3_SITELIB
        Python3_SOABI
    )
        unset(${_VAR} CACHE)
        unset(${_VAR})
    endforeach()
endmacro()
#
function(ROCPROFILER_SYSTEMS_PYBIND11_ADD_MODULE target_name)
    set(options EXCLUDE_FROM_ALL)
    set(args PYTHON_VERSION VISIBILITY CXX_STANDARD)
    set(kwargs)
    cmake_parse_arguments(ARG "${options}" "${args}" "${kwargs}" ${ARGN})

    if(NOT ARG_VISIBILITY)
        set(ARG_VISIBILITY "hidden")
    endif()
    if(NOT ARG_CXX_STANDARD)
        set(ARG_CXX_STANDARD ${CMAKE_CXX_STANDARD})
    endif()
    if(ARG_EXCLUDE_FROM_ALL)
        set(exclude_from_all EXCLUDE_FROM_ALL)
    endif()
    if(ARG_PYTHON_VERSION)
        set(_EXACT "EXACT")
    endif()

    # Relies on Python3_ROOT_DIR already being set in the caller's scope (e.g. the
    # per-version loop in source/python/CMakeLists.txt) to pin which interpreter is
    # found; intentionally not reset by rocprofiler_systems_reset_python3_cache() since
    # it is a find_package() hint, not a cached discovery output.
    set(Python3_FIND_STRATEGY "LOCATION")
    set(Python3_FIND_VIRTUALENV "FIRST")
    set(Python3_ARTIFACTS_INTERACTIVE OFF)

    rocprofiler_systems_reset_python3_cache()
    find_package(
        Python3
        ${ARG_PYTHON_VERSION}
        ${_EXACT}
        REQUIRED
        MODULE
        COMPONENTS Interpreter Development.Module
    )

    add_library(${target_name} MODULE ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})

    target_link_libraries(${target_name} PRIVATE pybind11::module)
    target_include_directories(${target_name} SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})

    set_target_properties(
        ${target_name}
        PROPERTIES
            PREFIX ""
            SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
            # -fvisibility=hidden is required to allow multiple modules compiled against
            # different pybind versions to work properly, and for some features (e.g.
            # py::module_local).
            CXX_VISIBILITY_PRESET "${ARG_VISIBILITY}"
            CXX_STANDARD ${ARG_CXX_STANDARD}
            CXX_STANDARD_REQUIRED ON
    )

    rocprofiler_systems_strip_target(${target_name} EXPLICIT)
endfunction()
