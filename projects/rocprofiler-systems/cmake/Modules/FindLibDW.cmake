# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#[=======================================================================[.rst:
FindLibDW
---------

Find libdw, the elfutils library for DWARF data and ELF file or process inspection.

This module is a thin compatibility shim. The canonical package name in this
project is the lowercase upstream ``libdw``; ``LibDW`` is the name Dyninst's
own ``FindElfutils.cmake`` invokes. This shim funnels both names through one
discovery path.

Discovery order:

1. If ``LibDW::LibDW`` already exists (e.g. created upfront by
   ``DyninstElfUtils.cmake`` for the from-source build), short-circuit.
2. ``find_package(libdw CONFIG)`` — picks up a vendored or generated
   ``libdw-config.cmake`` (TheRock sysdep, or the from-source build's
   generated config).
3. ``pkg_check_modules(libdw)`` — distro fallback for system installs.

Variables that affect this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LibDW_NO_SYSTEM_PATHS``
  If true, no system paths are searched (pkg-config fallback skipped).

Imported targets
^^^^^^^^^^^^^^^^

``LibDW::LibDW``
  The libdw library, if found.

Result variables
^^^^^^^^^^^^^^^^

``LibDW_FOUND``, ``LibDW_INCLUDE_DIRS``, ``LibDW_LIBRARIES``, ``LibDW_VERSION``
#]=======================================================================]

cmake_policy(SET CMP0074 NEW)

# 1. Short-circuit: already populated upstream (e.g. mode 1 pre-creates target).
if(TARGET LibDW::LibDW AND LibDW_LIBRARIES AND LibDW_INCLUDE_DIRS)
    set(LibDW_FOUND TRUE)
    return()
endif()

# 2. Try the canonical lowercase config package. Covers:
#    - TheRock sysdep (lib/rocm_sysdeps/lib/cmake/libdw/libdw-config.cmake)
#    - The from-source build's generated config (libdw_DIR pre-set)
if(NOT LibDW_FOUND)
    find_package(libdw ${LibDW_FIND_VERSION} CONFIG QUIET)
    if(libdw_FOUND AND TARGET libdw::libdw)
        set(LibDW_FOUND TRUE)
        set(LibDW_VERSION "${libdw_VERSION}")
        get_target_property(LibDW_INCLUDE_DIRS libdw::libdw INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(LibDW_LIBRARIES    libdw::libdw IMPORTED_LOCATION)
        if(NOT TARGET LibDW::LibDW)
            add_library(LibDW::LibDW INTERFACE IMPORTED)
            target_link_libraries(LibDW::LibDW INTERFACE libdw::libdw)
        endif()
    endif()
endif()

# 3. pkg-config fallback for system installs.
if(NOT LibDW_FOUND AND NOT LibDW_NO_SYSTEM_PATHS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        set(_version "")
        if(NOT "x${LibDW_FIND_VERSION}" STREQUAL "x")
            set(_version ">=${LibDW_FIND_VERSION}")
        endif()
        set(_quiet "")
        if(LibDW_FIND_QUIETLY)
            set(_quiet "QUIET")
        endif()
        pkg_check_modules(PC_LIBDW ${_quiet} "libdw${_version}")
        unset(_version)
        unset(_quiet)
    endif()

    if(PC_LIBDW_FOUND)
        if("x${PC_LIBDW_INCLUDE_DIRS}" STREQUAL "x")
            pkg_get_variable(PC_LIBDW_INCLUDE_DIRS libdw includedir)
        endif()
        set(LibDW_INCLUDE_DIRS "${PC_LIBDW_INCLUDE_DIRS}")
        set(LibDW_LIBRARIES    "${PC_LIBDW_LINK_LIBRARIES}")
        set(LibDW_VERSION      "${PC_LIBDW_VERSION}")

        # Some platforms list libelf as a dependency; split it off so the
        # imported target's IMPORTED_LOCATION points only at libdw.
        list(LENGTH LibDW_LIBRARIES _cnt)
        if(${_cnt} GREATER 1)
            set(_libdw "")
            set(_link_libs "")
            foreach(_l ${LibDW_LIBRARIES})
                if(${_l} MATCHES "libdw")
                    set(_libdw ${_l})
                else()
                    list(APPEND _link_libs ${_l})
                endif()
            endforeach()
            if(_libdw)
                set(LibDW_LIBRARIES ${_libdw})
            endif()
        endif()
        unset(_cnt)

        if(NOT TARGET LibDW::LibDW)
            add_library(LibDW::LibDW UNKNOWN IMPORTED)
            set_target_properties(LibDW::LibDW PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${LibDW_LIBRARIES}"
                INTERFACE_INCLUDE_DIRECTORIES "${LibDW_INCLUDE_DIRS}"
            )
            if(_link_libs)
                set_target_properties(LibDW::LibDW PROPERTIES
                    IMPORTED_LINK_DEPENDENT_LIBRARIES "${_link_libs}"
                )
            endif()
        endif()
        unset(_libdw)
        unset(_link_libs)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibDW
    FOUND_VAR LibDW_FOUND
    REQUIRED_VARS LibDW_LIBRARIES LibDW_INCLUDE_DIRS
    VERSION_VAR LibDW_VERSION
)

if(LibDW_FOUND)
    mark_as_advanced(LibDW_INCLUDE_DIRS LibDW_LIBRARIES LibDW_VERSION)
endif()
