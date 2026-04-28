# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#[=======================================================================[.rst:
FindLibUnwind
-------------

Find libunwind, the library for unwinding call stacks at runtime.

Variables that affect this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LibUnwind_NO_SYSTEM_PATHS``
  If `True`, no system paths are searched.

``LibUnwind_ROOT``
  Root directory to search for libunwind.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``LibUnwind::LibUnwind``
  The libunwind library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``LibUnwind_INCLUDE_DIRS``
  where to find libunwind.h, etc.
``LibUnwind_LIBRARIES``
  the libraries to link against to use libunwind.
``LibUnwind_FOUND``
  If false, do not try to use libunwind.
``LibUnwind_VERSION``
  the version of the libunwind library found

#]=======================================================================]
cmake_policy(SET CMP0074 NEW) # Use <Package>_ROOT

if(LibUnwind_NO_SYSTEM_PATHS)
    set(_find_path_args NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
endif()

# There is no way to tell pkg-config to ignore directories, so disable it
if(NOT LibUnwind_NO_SYSTEM_PATHS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        if(NOT "x${LibUnwind_FIND_VERSION}" STREQUAL "x")
            set(_version ">=${LibUnwind_FIND_VERSION}")
        endif()
        if(LibUnwind_FIND_QUIETLY)
            set(_quiet "QUIET")
        endif()

        pkg_check_modules(PC_LIBUNWIND ${_quiet} "libunwind${_version}")
        unset(_version)
        unset(_quiet)
    endif()
endif()

if(PC_LIBUNWIND_FOUND)
    if("x${PC_LIBUNWIND_INCLUDE_DIRS}" STREQUAL "x")
        pkg_get_variable(PC_LIBUNWIND_INCLUDE_DIRS libunwind includedir)
    endif()

    set(LibUnwind_INCLUDE_DIRS ${PC_LIBUNWIND_INCLUDE_DIRS} CACHE PATH "")
    set(LibUnwind_LIBRARIES ${PC_LIBUNWIND_LINK_LIBRARIES} CACHE PATH "")
    set(LibUnwind_VERSION ${PC_LIBUNWIND_VERSION} CACHE STRING "")
else()
    find_path(
        LibUnwind_INCLUDE_DIRS
        NAMES libunwind.h
        HINTS ${LibUnwind_ROOT}
        PATH_SUFFIXES include ${_find_path_args}
    )

    find_library(
        LibUnwind_LIBRARIES
        NAMES libunwind unwind
        HINTS ${LibUnwind_ROOT}
        PATH_SUFFIXES lib lib64 ${_find_path_args}
    )

    if(LibUnwind_INCLUDE_DIRS AND EXISTS "${LibUnwind_INCLUDE_DIRS}/libunwind-common.h")
        file(
            STRINGS "${LibUnwind_INCLUDE_DIRS}/libunwind-common.h"
            _version_line
            REGEX "^#define UNW_VERSION_(MAJOR|MINOR|EXTRA)[ \t]+[0-9]+"
        )
        string(REGEX MATCH "MAJOR[ \t]+([0-9]+)" _major "${_version_line}")
        string(REGEX MATCH "MINOR[ \t]+([0-9]+)" _minor "${_version_line}")
        if(_major AND _minor)
            string(REGEX REPLACE ".*([0-9]+)$" "\\1" _major_num "${_major}")
            string(REGEX REPLACE ".*([0-9]+)$" "\\1" _minor_num "${_minor}")
            set(LibUnwind_VERSION "${_major_num}.${_minor_num}")
        endif()
        unset(_version_line)
        unset(_major)
        unset(_minor)
        unset(_major_num)
        unset(_minor_num)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LibUnwind
    FOUND_VAR LibUnwind_FOUND
    REQUIRED_VARS LibUnwind_LIBRARIES LibUnwind_INCLUDE_DIRS
    VERSION_VAR LibUnwind_VERSION
)

if(LibUnwind_FOUND)
    mark_as_advanced(LibUnwind_INCLUDE_DIRS)
    mark_as_advanced(LibUnwind_LIBRARIES)
    mark_as_advanced(LibUnwind_VERSION)

    if(NOT TARGET LibUnwind::LibUnwind)
        add_library(LibUnwind::LibUnwind UNKNOWN IMPORTED)
        set_target_properties(
            LibUnwind::LibUnwind
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${LibUnwind_INCLUDE_DIRS}"
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${LibUnwind_LIBRARIES}"
        )
    endif()
endif()

unset(_find_path_args)
