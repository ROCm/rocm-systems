# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# =======================================================================================
#
# ConfigureDyninstTargets.cmake
#
# This module configures Dyninst to use the dependencies we've found or built.
#
# Strategy:
# - If we're building a dependency (ROCPROFSYS_BUILD_* = ON), we create the Dyninst::*
#   target manually pointing to our rocprofiler-systems-* target. This prevents Dyninst
#   from trying to find_package() a library that won't exist until build time.
# - If we found a system dependency (ROCPROFSYS_BUILD_* = OFF), we set the *_ROOT_DIR
#   variables so Dyninst's find_package() calls locate the same system package we found.
#
# This file must be included AFTER DyninstExternals.cmake (which builds/finds the deps)
# and BEFORE add_subdirectory(external/dyninst) in Packages.cmake.
#
# =======================================================================================

include_guard(GLOBAL)

# --------------------------------------------------------------------------------------#
# Boost
# --------------------------------------------------------------------------------------#
if(ROCPROFSYS_BUILD_BOOST)
    # We're building Boost from source - it won't be available until build time.
    # Create the Dyninst::Boost target manually to bypass Dyninst's find_package.
    if(NOT TARGET Dyninst::Boost AND TARGET rocprofiler-systems-boost)
        add_library(Dyninst::Boost INTERFACE IMPORTED)
        set_target_properties(
            Dyninst::Boost
            PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-boost
        )
        # Also create the headers-only target if it doesn't exist
        if(NOT TARGET Dyninst::Boost_headers)
            add_library(Dyninst::Boost_headers INTERFACE IMPORTED)
            set_target_properties(
                Dyninst::Boost_headers
                PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-boost
            )
        endif()
        rocprofiler_systems_message(
            STATUS
                "Created Dyninst::Boost target linked to rocprofiler-systems-boost (building from source)"
        )
    endif()
else()
    # We found system Boost - configure Dyninst to find the same one.
    if(Boost_FOUND)
        set(Boost_ROOT_DIR
            "${Boost_ROOT_DIR}"
            CACHE PATH
            "Boost root directory for Dyninst"
            FORCE
        )
        # If we're using a non-system location, tell Dyninst not to search system paths
        if(Boost_ROOT_DIR AND NOT Boost_ROOT_DIR STREQUAL "/usr")
            set(Boost_NO_SYSTEM_PATHS
                ON
                CACHE BOOL
                "Disable system path search for Boost in Dyninst"
                FORCE
            )
        endif()
        rocprofiler_systems_message(
            STATUS "Configured Dyninst to find system Boost from: ${Boost_ROOT_DIR}"
        )
    endif()
endif()

# --------------------------------------------------------------------------------------#
# TBB
# --------------------------------------------------------------------------------------#
if(ROCPROFSYS_BUILD_TBB)
    # We're building TBB from source - create target manually.
    if(NOT TARGET Dyninst::TBB AND TARGET rocprofiler-systems-tbb)
        add_library(Dyninst::TBB INTERFACE IMPORTED)
        set_target_properties(
            Dyninst::TBB
            PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-tbb
        )
        rocprofiler_systems_message(
            STATUS
                "Created Dyninst::TBB target linked to rocprofiler-systems-tbb (building from source)"
        )
    endif()
else()
    # We found system TBB - configure Dyninst to find it.
    if(TBB_FOUND)
        set(TBB_ROOT "${TBB_ROOT_DIR}" CACHE PATH "TBB root directory for Dyninst" FORCE)
        set(TBB_ROOT_DIR
            "${TBB_ROOT_DIR}"
            CACHE PATH
            "TBB root directory for Dyninst"
            FORCE
        )
        rocprofiler_systems_message(
            STATUS "Configured Dyninst to find system TBB from: ${TBB_ROOT_DIR}"
        )
    endif()
endif()

# --------------------------------------------------------------------------------------#
# ElfUtils
# --------------------------------------------------------------------------------------#
if(ROCPROFSYS_BUILD_ELFUTILS)
    # We're building ElfUtils from source - create target manually.
    if(NOT TARGET Dyninst::ElfUtils AND TARGET rocprofiler-systems-elfutils)
        add_library(Dyninst::ElfUtils INTERFACE IMPORTED)
        set_target_properties(
            Dyninst::ElfUtils
            PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-elfutils
        )
        rocprofiler_systems_message(
            STATUS
                "Created Dyninst::ElfUtils target linked to rocprofiler-systems-elfutils (building from source)"
        )
    endif()
else()
    # We found system ElfUtils - configure Dyninst to find it.
    if(LibElf_FOUND OR LibDwarf_FOUND)
        set(ElfUtils_ROOT_DIR
            "${ElfUtils_ROOT_DIR}"
            CACHE PATH
            "ElfUtils root directory for Dyninst"
            FORCE
        )
        rocprofiler_systems_message(
            STATUS "Configured Dyninst to find system ElfUtils from: ${ElfUtils_ROOT_DIR}"
        )
    endif()
endif()

# --------------------------------------------------------------------------------------#
# LibIberty
# --------------------------------------------------------------------------------------#
if(ROCPROFSYS_BUILD_LIBIBERTY)
    # We're building LibIberty from source - create target manually.
    if(NOT TARGET Dyninst::LibIberty AND TARGET rocprofiler-systems-libiberty)
        add_library(Dyninst::LibIberty INTERFACE IMPORTED)
        set_target_properties(
            Dyninst::LibIberty
            PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-libiberty
        )
        rocprofiler_systems_message(
            STATUS
                "Created Dyninst::LibIberty target linked to rocprofiler-systems-libiberty (building from source)"
        )
    endif()
else()
    # We found system LibIberty - configure Dyninst to find it.
    if(LibIberty_FOUND)
        set(LibIberty_ROOT_DIR
            "${LibIberty_ROOT_DIR}"
            CACHE PATH
            "LibIberty root directory for Dyninst"
            FORCE
        )
        rocprofiler_systems_message(
            STATUS
                "Configured Dyninst to find system LibIberty from: ${LibIberty_ROOT_DIR}"
        )
    endif()
endif()
