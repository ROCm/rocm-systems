# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(FMT_VERSION "11.1.3" CACHE STRING "Minimum fmt version")

if(PROFILER_HUB_FETCH_DEPENDENCIES)
    include(FetchContent)

    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG ${FMT_VERSION}
        GIT_SHALLOW TRUE
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

    FetchContent_MakeAvailable(fmt)

    set(BUILD_SHARED_LIBS ${_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP})
    unset(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP)

    if(TARGET fmt)
        set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    if(NOT TARGET fmt::fmt)
        add_library(fmt::fmt ALIAS fmt)
    endif()
else()
    find_package(fmt ${FMT_VERSION})

    if(NOT fmt_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires fmt ${FMT_VERSION} or newer on CMAKE_PREFIX_PATH. Configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it instead."
        )
    endif()

    message(STATUS "Using system fmt (version ${fmt_VERSION})")
endif()
