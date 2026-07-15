# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

set(ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY
    "https://github.com/ROCm/rocm-systems.git"
    CACHE STRING
    "Git repository URL for profiler-hub fallback sparse checkout"
)

set(ROCPROFSYS_PROFILER_HUB_GIT_TAG
    "users/adjordje-amd/profiler-hub-fix-fmt-export"
    CACHE STRING
    "Git tag/branch for profiler-hub fallback sparse checkout"
)

set(ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR
    "profilers/profiler-hub"
    CACHE STRING
    "Subdirectory inside the repository that contains profiler-hub"
)

option(ROCPROFSYS_PROFILER_HUB_ENABLE_LOGGING "Enable profiler-hub logging" OFF)
option(ROCPROFSYS_PROFILER_HUB_LINK_STATIC "Link profiler-hub statically" OFF)

if(ROCPROFSYS_PROFILER_HUB_LINK_STATIC)
    set(_PROFILER_HUB_SUFFIX "-static")
    message(STATUS "[profiler-hub] Linking statically")
else()
    set(_PROFILER_HUB_SUFFIX "")
endif()

# The imported-target shape (profiler-hub::profiler-hub[-static]) is identical
# whether profiler-hub comes from an installed package (tier 1) or is fetched
# and built as an ExternalProject (tier 2), so downstream code never needs to
# branch on which tier produced it.
set(_PROFILER_HUB_TARGET "profiler-hub::profiler-hub${_PROFILER_HUB_SUFFIX}")

# ------------------------------------------------------------------------------
# Resolution order:
#   1. find_package(profiler-hub) - use installed package if present
#   2. fallback: fetch+build ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR from
#      ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY at ROCPROFSYS_PROFILER_HUB_GIT_TAG
#      as a build-time ExternalProject (sparse checkout, own cmake configure/
#      build/install), so the fetch does not block configure and Ninja/Make
#      can schedule it alongside unrelated targets.
# ------------------------------------------------------------------------------

find_package(profiler-hub QUIET CONFIG)

if(profiler-hub_FOUND)
    message(STATUS "[profiler-hub] Using installed package: ${profiler-hub_DIR}")
else()
    message(
        STATUS
        "[profiler-hub] find_package failed; falling back to a build-time fetch of "
        "${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY} (${ROCPROFSYS_PROFILER_HUB_GIT_TAG})"
    )

    find_package(Git REQUIRED)

    set(_PROFILER_HUB_ROOT "${PROJECT_BINARY_DIR}/external/profiler-hub")
    set(_PROFILER_HUB_STAMP "${_PROFILER_HUB_ROOT}/.checkout.stamp")
    set(_PROFILER_HUB_INSTALL_DIR "${_PROFILER_HUB_ROOT}/install")

    # profiler-hub's own `project(profiler-hub VERSION ...)` (see
    # profilers/profiler-hub/CMakeLists.txt) is the single source of truth for
    # this. ExternalProject's BUILD_BYPRODUCTS/IMPORTED_LOCATION need the exact
    # installed filenames *before* the fetch has happened, so the version has
    # to be duplicated here; bump it if profiler-hub's own version changes.
    set(_PROFILER_HUB_VERSION "0.1.0")
    set(_PROFILER_HUB_SOVERSION "0")

    set(_PROFILER_HUB_SHARED_LIB
        "${_PROFILER_HUB_INSTALL_DIR}/lib/libprofiler-hub${CMAKE_SHARED_LIBRARY_SUFFIX}.${_PROFILER_HUB_VERSION}"
    )
    set(_PROFILER_HUB_SHARED_SONAME
        "${_PROFILER_HUB_INSTALL_DIR}/lib/libprofiler-hub${CMAKE_SHARED_LIBRARY_SUFFIX}.${_PROFILER_HUB_SOVERSION}"
    )
    set(_PROFILER_HUB_SHARED_LINK
        "${_PROFILER_HUB_INSTALL_DIR}/lib/libprofiler-hub${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
    set(_PROFILER_HUB_STATIC_LIB
        "${_PROFILER_HUB_INSTALL_DIR}/lib/libprofiler-hub${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )

    # CMAKE_PREFIX_PATH is a ;-separated list; ExternalProject_Add's CMAKE_ARGS
    # entries are themselves list items, so the inner list separator has to be
    # escaped to something else and un-escaped via LIST_SEPARATOR, otherwise
    # CMake mis-splits the argument.
    string(REPLACE ";" "|" _PROFILER_HUB_PREFIX_PATH_ESCAPED "${CMAKE_PREFIX_PATH}")

    # profiler-hub's own cmake/fmt.cmake does find_package(fmt 11.2.0 QUIET)
    # against the same CMAKE_PREFIX_PATH we forward below, then only exports
    # fmt::fmt as a real (non-BUILD_INTERFACE) link requirement of
    # profiler-hub-static when that find_package succeeds; otherwise fmt is
    # vendored and kept build-only, invisible to consumers. Since the nested
    # build's own package config doesn't exist yet at our configure time, this
    # detection has to be duplicated here so the hand-declared IMPORTED target
    # below carries the same link requirement the real installed package would.
    set(_PROFILER_HUB_FMT_VERSION "11.2.0")
    find_package(fmt ${_PROFILER_HUB_FMT_VERSION} QUIET)

    include(ExternalProject)
    ExternalProject_Add(
        profiler_hub_external
        PREFIX "${_PROFILER_HUB_ROOT}"
        SOURCE_DIR "${_PROFILER_HUB_ROOT}/src"
        BINARY_DIR "${_PROFILER_HUB_ROOT}/build"
        INSTALL_DIR "${_PROFILER_HUB_INSTALL_DIR}"
        # STAMP_DIR/TMP_DIR default to a location derived from SOURCE_DIR's name
        # ("src"), which would otherwise nest ExternalProject's own per-step
        # bookkeeping files inside SOURCE_DIR itself - exactly what the sparse
        # checkout script wipes with REMOVE_RECURSE on every re-checkout.
        STAMP_DIR "${_PROFILER_HUB_ROOT}/stamp"
        TMP_DIR "${_PROFILER_HUB_ROOT}/tmp"
        LIST_SEPARATOR "|"
        # ExternalProject runs a custom DOWNLOAD_COMMAND with <SOURCE_DIR> itself
        # as the process's working directory, but the script below removes and
        # recreates <SOURCE_DIR> as its first action - deleting your own cwd
        # invalidates it for the rest of the process. `cmake -E chdir` forces the
        # actual cmake -P process to start from this file's (never-deleted)
        # directory instead, so it never runs with a stale/removed cwd.
        DOWNLOAD_COMMAND
            ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_COMMAND}
            -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
            -DREPO_URL=${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY}
            -DGIT_TAG=${ROCPROFSYS_PROFILER_HUB_GIT_TAG}
            -DGIT_SUBDIR=${ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR} -DCHECKOUT_DIR=<SOURCE_DIR>
            -DSTAMP_FILE=${_PROFILER_HUB_STAMP} -P
            ${CMAKE_CURRENT_LIST_DIR}/scripts/SparseCheckoutProfilerHub.cmake
        UPDATE_COMMAND ""
        SOURCE_SUBDIR "${ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR}"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_PREFIX_PATH=${_PROFILER_HUB_PREFIX_PATH_ESCAPED}
            -DCMAKE_INSTALL_LIBDIR=lib -DPROFILER_HUB_BUILD_TESTS=OFF
            -DPROFILER_HUB_BUILD_BENCHMARKS=OFF
            -DPROFILER_HUB_ENABLE_LOGGING=${ROCPROFSYS_PROFILER_HUB_ENABLE_LOGGING}
        BUILD_BYPRODUCTS
            "${_PROFILER_HUB_SHARED_LIB}"
            "${_PROFILER_HUB_SHARED_SONAME}"
            "${_PROFILER_HUB_SHARED_LINK}"
            "${_PROFILER_HUB_STATIC_LIB}"
        EXCLUDE_FROM_ALL TRUE
    )

    file(MAKE_DIRECTORY "${_PROFILER_HUB_INSTALL_DIR}/include")

    add_library(profiler-hub::profiler-hub SHARED IMPORTED GLOBAL)
    set_target_properties(
        profiler-hub::profiler-hub
        PROPERTIES
            IMPORTED_LOCATION "${_PROFILER_HUB_SHARED_LIB}"
            IMPORTED_SONAME
                "libprofiler-hub${CMAKE_SHARED_LIBRARY_SUFFIX}.${_PROFILER_HUB_SOVERSION}"
            INTERFACE_INCLUDE_DIRECTORIES "${_PROFILER_HUB_INSTALL_DIR}/include"
    )

    add_library(profiler-hub::profiler-hub-static STATIC IMPORTED GLOBAL)
    set_target_properties(
        profiler-hub::profiler-hub-static
        PROPERTIES
            IMPORTED_LOCATION "${_PROFILER_HUB_STATIC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${_PROFILER_HUB_INSTALL_DIR}/include"
    )
    if(fmt_FOUND)
        set_property(
            TARGET profiler-hub::profiler-hub-static
            APPEND
            PROPERTY INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:fmt::fmt>"
        )
    endif()

    unset(_PROFILER_HUB_PREFIX_PATH_ESCAPED)
endif()

# ------------------------------------------------------------------------------
# Interface target
# ------------------------------------------------------------------------------

add_library(rocprofiler-systems-profiler-hub INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-profiler-hub
    ALIAS rocprofiler-systems-profiler-hub
)
target_link_libraries(rocprofiler-systems-profiler-hub INTERFACE ${_PROFILER_HUB_TARGET})

if(TARGET profiler_hub_external)
    add_dependencies(rocprofiler-systems-profiler-hub profiler_hub_external)
endif()
