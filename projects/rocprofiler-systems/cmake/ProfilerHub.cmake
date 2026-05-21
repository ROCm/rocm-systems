# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(
    ROCPROFSYS_USE_EXTERNAL_PROFILER_HUB
    "Use externally installed profiler-hub library instead of building from source"
    OFF
)

set(ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY
    ""
    CACHE STRING
    "Git repository URL for profiler-hub (leave empty to use local path)"
)

set(ROCPROFSYS_PROFILER_HUB_GIT_TAG
    "develop"
    CACHE STRING
    "Git tag/branch for profiler-hub"
)

set(ROCPROFSYS_PROFILER_HUB_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/../../profilers/profiler-hub"
    CACHE PATH
    "Local path to profiler-hub source directory"
)

option(ROCPROFSYS_PROFILER_HUB_ENABLE_LOGGING "Enable profiler-hub logging" OFF)
option(ROCPROFSYS_PROFILER_HUB_LINK_STATIC "Link profiler-hub statically" OFF)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

if(ROCPROFSYS_USE_EXTERNAL_PROFILER_HUB)
    find_package(profiler-hub REQUIRED)
    message(STATUS "[profiler-hub] Using external installation: ${profiler-hub_DIR}")

    set(_PROFILER_HUB_IS_EXTERNAL TRUE)
else()
    include(FetchContent)

    set(PROFILER_HUB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PROFILER_HUB_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(PROFILER_HUB_ENABLE_LOGGING
        ${ROCPROFSYS_PROFILER_HUB_ENABLE_LOGGING}
        CACHE BOOL
        ""
        FORCE
    )

    if(ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY)
        message(
            STATUS
            "[profiler-hub] Fetching from Git: ${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY} (${ROCPROFSYS_PROFILER_HUB_GIT_TAG})"
        )
        set(_PROFILER_HUB_SOURCE_ARGS
            GIT_REPOSITORY
            ${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY}
            GIT_TAG
            ${ROCPROFSYS_PROFILER_HUB_GIT_TAG}
            SOURCE_DIR
            ${PROJECT_BINARY_DIR}/external/profiler-hub/src
        )
    else()
        if(NOT EXISTS "${ROCPROFSYS_PROFILER_HUB_SOURCE_DIR}/CMakeLists.txt")
            message(
                FATAL_ERROR
                "[profiler-hub] Local source not found at ${ROCPROFSYS_PROFILER_HUB_SOURCE_DIR}. "
                "Please set ROCPROFSYS_PROFILER_HUB_SOURCE_DIR to a valid profiler-hub source "
                "directory, or set ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY to fetch from git."
            )
        else()
            message(
                STATUS
                "[profiler-hub] Using local source: ${ROCPROFSYS_PROFILER_HUB_SOURCE_DIR}"
            )
            set(_PROFILER_HUB_SOURCE_ARGS
                SOURCE_DIR
                ${ROCPROFSYS_PROFILER_HUB_SOURCE_DIR}
            )
        endif()
    endif()

    FetchContent_Declare(
        profiler-hub
        ${_PROFILER_HUB_SOURCE_ARGS}
        BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/profiler-hub/build
        SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/profiler-hub/subbuild
    )
    FetchContent_MakeAvailable(profiler-hub)

    set(_PROFILER_HUB_IS_EXTERNAL FALSE)
endif()

if(ROCPROFSYS_PROFILER_HUB_LINK_STATIC)
    set(_PROFILER_HUB_SUFFIX "-static")
    message(STATUS "[profiler-hub] Linking statically")
else()
    set(_PROFILER_HUB_SUFFIX "")
endif()

if(_PROFILER_HUB_IS_EXTERNAL)
    set(_PROFILER_HUB_TARGET profiler-hub::profiler-hub${_PROFILER_HUB_SUFFIX})
else()
    set(_PROFILER_HUB_TARGET profiler-hub${_PROFILER_HUB_SUFFIX})
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

if(NOT _PROFILER_HUB_IS_EXTERNAL)
    target_include_directories(
        rocprofiler-systems-profiler-hub
        SYSTEM
        INTERFACE $<TARGET_PROPERTY:profiler-hub,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
