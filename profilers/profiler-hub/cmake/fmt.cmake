# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(FMT_VERSION "11.1.3" CACHE STRING "Minimum fmt version")

# Fetching is On by default: this branch runs only under
# -DPROFILER_HUB_FETCH_DEPENDENCIES=OFF.
if(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    find_package(fmt ${FMT_VERSION})

    if(NOT fmt_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires fmt ${FMT_VERSION} or newer on CMAKE_PREFIX_PATH. Drop -DPROFILER_HUB_FETCH_DEPENDENCIES=OFF to download it instead."
        )
    endif()

    message(STATUS "Using system fmt (version ${fmt_VERSION})")
else()
    include(FetchContent)

    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG ${FMT_VERSION}
        GIT_SHALLOW TRUE
        # Without this, the MakeAvailable below always fetches.
        FIND_PACKAGE_ARGS ${FMT_VERSION}
    )

    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    set(FMT_DOC OFF CACHE BOOL "" FORCE)
    set(FMT_TEST OFF CACHE BOOL "" FORCE)

    # Force a static, PIC fmt regardless of a parent project's BUILD_SHARED_LIBS
    # (e.g. rocprofiler-systems sets it ON project-wide, which would otherwise
    # produce a libfmt.so that FMT_INSTALL=OFF never installs, leaving it
    # unresolvable at runtime).
    set(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    # Tries find_package() first, fetches only if that fails.
    FetchContent_MakeAvailable(fmt)

    set(BUILD_SHARED_LIBS ${_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP})
    unset(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP)

    if(TARGET fmt)
        set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    if(NOT TARGET fmt::fmt)
        add_library(fmt::fmt ALIAS fmt)
    endif()
endif()
