# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ========================================================================================
# DyninstExternals.cmake
#
# Orchestrates the configuration and building of Dyninst's external dependencies
#
# ----------------------------------------
#
# This module:
# - Maps deprecated DYNINST_BUILD_* variables to ROCPROFSYS_BUILD_* variables
# - Sets up TPL_STAGING_PREFIX for third-party library installation
# - Includes and configures Dyninst dependencies (Boost, TBB, ElfUtils, LibIberty)
# - Creates build targets with serialized dependency chains
# - Creates external-prebuild and external-deps-complete targets for coordination
#
# ========================================================================================

include(MacroUtilities)

# Map deprecated DYNINST_BUILD_* variables to new ROCPROFSYS_BUILD_* variables
foreach(dep BOOST TBB ELFUTILS LIBIBERTY)
    if(DYNINST_BUILD_${dep})
        message(
            WARNING
            "DYNINST_BUILD_${dep} is deprecated. Using ROCPROFSYS_BUILD_${dep} instead."
        )
        set(ROCPROFSYS_BUILD_${dep} ON)
    endif()
endforeach()

set(TPL_STAGING_PREFIX
    "${PROJECT_BINARY_DIR}/external"
    CACHE PATH
    "Third-party library build-tree install prefix"
)
file(MAKE_DIRECTORY "${TPL_STAGING_PREFIX}")
file(MAKE_DIRECTORY "${TPL_STAGING_PREFIX}/include")

add_custom_target(external-prebuild)

# Add external dependencies to be built
include(DyninstBoost)
if(TARGET rocprofiler-systems-boost-build)
    # Make Boost build serially
    set_target_properties(
        rocprofiler-systems-boost
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    # Create a prebuild target that depends on Boost
    add_dependencies(external-prebuild rocprofiler-systems-boost-build)
endif()

include(DyninstTBB)
if(TARGET rocprofiler-systems-tbb-build AND TARGET external-prebuild)
    # Make TBB build serially and wait for Boost
    set_target_properties(
        rocprofiler-systems-tbb-build
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    add_dependencies(external-prebuild rocprofiler-systems-tbb-build)
endif()

include(DyninstElfUtils)
if(TARGET rocprofiler-systems-elfutils-build AND TARGET external-prebuild)
    set_target_properties(
        rocprofiler-systems-elfutils-build
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    add_dependencies(external-prebuild rocprofiler-systems-elfutils-build)
endif()

include(DyninstLibIberty)
if(TARGET rocprofiler-systems-libiberty-build AND TARGET external-prebuild)
    set_target_properties(
        rocprofiler-systems-libiberty-build
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    if(TARGET rocprofiler-systems-libiberty-install)
        add_dependencies(external-prebuild rocprofiler-systems-libiberty-install)
    else()
        add_dependencies(external-prebuild rocprofiler-systems-libiberty-build)
    endif()
endif()

# Final dependency check
if(NOT TARGET external-prebuild)
    message(WARNING "Not all dyninst external dependencies found. Build may fail.")
endif()

# Create a dummy target to ensure external dependencies are fully built
add_custom_target(external-deps-complete)
if(TARGET external-prebuild)
    add_dependencies(external-deps-complete external-prebuild)
endif()
