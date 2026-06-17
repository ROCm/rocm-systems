# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#[=======================================================================[.rst:
FindLibElf
----------

Find libelf, the elfutils library for ELF file inspection.

This module mirrors :module:`FindLibDW`: a thin shim that funnels discovery
through pkg-config, which covers both system installs and the TheRock
vendored sysdep.

Discovery order:

1. If ``LibElf::LibElf`` already exists (e.g. pre-created by
   ``DyninstElfUtils.cmake`` for the from-source build), short-circuit.
2. ``pkg_check_modules(libelf)`` — canonical path for both system
   installs (``/usr``) and the TheRock vendored sysdep (which ships
   ``libelf.pc`` in ``lib/rocm_sysdeps/lib/pkgconfig/``).

Imported targets
^^^^^^^^^^^^^^^^

``LibElf::LibElf``
  The libelf library, if found.

Result variables
^^^^^^^^^^^^^^^^

``LibElf_FOUND``, ``LibElf_INCLUDE_DIRS``, ``LibElf_LIBRARIES``, ``LibElf_VERSION``
#]=======================================================================]

# 1. Short-circuit: already populated upstream.
if(TARGET LibElf::LibElf AND LibElf_LIBRARIES AND LibElf_INCLUDE_DIRS)
    set(LibElf_FOUND TRUE)
    return()
endif()

# 2. pkg-config: the canonical path for both system installs (/usr) and
#    the TheRock vendored sysdep (lib/rocm_sysdeps/lib/pkgconfig/libelf.pc).
if(NOT LibElf_FOUND AND NOT LibElf_NO_SYSTEM_PATHS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        set(_version "")
        if(NOT "x${LibElf_FIND_VERSION}" STREQUAL "x")
            set(_version ">=${LibElf_FIND_VERSION}")
        endif()
        set(_quiet "")
        if(LibElf_FIND_QUIETLY)
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
        set(LibElf_INCLUDE_DIRS "${PC_LIBELF_INCLUDE_DIRS}")
        set(LibElf_LIBRARIES "${PC_LIBELF_LINK_LIBRARIES}")
        set(LibElf_VERSION "${PC_LIBELF_VERSION}")

        if(NOT TARGET LibElf::LibElf)
            add_library(LibElf::LibElf UNKNOWN IMPORTED)
            set_target_properties(
                LibElf::LibElf
                PROPERTIES
                    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                    IMPORTED_LOCATION "${LibElf_LIBRARIES}"
                    INTERFACE_INCLUDE_DIRECTORIES "${LibElf_INCLUDE_DIRS}"
            )
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LibElf
    FOUND_VAR LibElf_FOUND
    REQUIRED_VARS LibElf_LIBRARIES LibElf_INCLUDE_DIRS
    VERSION_VAR LibElf_VERSION
)

if(LibElf_FOUND)
    mark_as_advanced(LibElf_INCLUDE_DIRS LibElf_LIBRARIES LibElf_VERSION)
endif()
