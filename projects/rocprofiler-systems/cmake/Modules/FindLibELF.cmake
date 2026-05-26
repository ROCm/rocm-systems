# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#[=======================================================================[.rst:
FindLibELF
----------

Find libelf, the elfutils library for ELF file inspection.

This module mirrors :module:`FindLibDW`: a thin shim that funnels discovery
through the lowercase upstream ``libelf`` config package first, then falls
back to pkg-config for distro installs.

Named ``FindLibELF`` (uppercase ELF) to match the contract Dyninst's
``FindElfutils.cmake`` calls — ``find_package(LibELF)``. Linux is
case-sensitive about the filename.

Discovery order:

1. If ``LibELF::LibELF`` already exists (e.g. pre-created by
   ``DyninstElfUtils.cmake`` for the from-source build), short-circuit.
2. ``find_package(libelf CONFIG)`` — vendored or generated config.
3. ``pkg_check_modules(libelf)`` — distro fallback.

Imported targets
^^^^^^^^^^^^^^^^

``LibELF::LibELF``
  The libelf library, if found.

Result variables
^^^^^^^^^^^^^^^^

``LibELF_FOUND``, ``LibELF_INCLUDE_DIRS``, ``LibELF_LIBRARIES``, ``LibELF_VERSION``
#]=======================================================================]

cmake_policy(SET CMP0074 NEW)

# 1. Short-circuit: already populated upstream.
if(TARGET LibELF::LibELF AND LibELF_LIBRARIES AND LibELF_INCLUDE_DIRS)
    set(LibELF_FOUND TRUE)
    return()
endif()

# 2. Try the canonical lowercase config package.
if(NOT LibELF_FOUND)
    find_package(libelf ${LibELF_FIND_VERSION} CONFIG QUIET)
    if(libelf_FOUND AND TARGET libelf::libelf)
        set(LibELF_FOUND TRUE)
        set(LibELF_VERSION "${libelf_VERSION}")
        get_target_property(LibELF_INCLUDE_DIRS libelf::libelf INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(LibELF_LIBRARIES    libelf::libelf IMPORTED_LOCATION)
        if(NOT TARGET LibELF::LibELF)
            add_library(LibELF::LibELF INTERFACE IMPORTED)
            target_link_libraries(LibELF::LibELF INTERFACE libelf::libelf)
        endif()
    endif()
endif()

# 3. pkg-config fallback for system installs.
if(NOT LibELF_FOUND AND NOT LibELF_NO_SYSTEM_PATHS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        set(_version "")
        if(NOT "x${LibELF_FIND_VERSION}" STREQUAL "x")
            set(_version ">=${LibELF_FIND_VERSION}")
        endif()
        set(_quiet "")
        if(LibELF_FIND_QUIETLY)
            set(_quiet "QUIET")
        endif()
        pkg_check_modules(PC_LIBELF ${_quiet} "libelf${_version}")
        unset(_version)
        unset(_quiet)
    endif()

    if(PC_LIBELF_FOUND)
        if("x${PC_LIBELF_INCLUDE_DIRS}" STREQUAL "x")
            pkg_get_variable(PC_LIBELF_INCLUDE_DIRS libelf includedir)
        endif()
        set(LibELF_INCLUDE_DIRS "${PC_LIBELF_INCLUDE_DIRS}")
        set(LibELF_LIBRARIES    "${PC_LIBELF_LINK_LIBRARIES}")
        set(LibELF_VERSION      "${PC_LIBELF_VERSION}")

        if(NOT TARGET LibELF::LibELF)
            add_library(LibELF::LibELF UNKNOWN IMPORTED)
            set_target_properties(LibELF::LibELF PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${LibELF_LIBRARIES}"
                INTERFACE_INCLUDE_DIRECTORIES "${LibELF_INCLUDE_DIRS}"
            )
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibELF
    FOUND_VAR LibELF_FOUND
    REQUIRED_VARS LibELF_LIBRARIES LibELF_INCLUDE_DIRS
    VERSION_VAR LibELF_VERSION
)

if(LibELF_FOUND)
    mark_as_advanced(LibELF_INCLUDE_DIRS LibELF_LIBRARIES LibELF_VERSION)
endif()
