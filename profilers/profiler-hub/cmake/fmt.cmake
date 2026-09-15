# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(FMT_VERSION "11.1.3" CACHE STRING "Minimum fmt version")

find_package(fmt QUIET)

set(_fmt_reason "")
if(NOT fmt_FOUND)
    set(_fmt_reason "no package was found on CMAKE_PREFIX_PATH")
elseif(fmt_VERSION VERSION_LESS FMT_VERSION)
    set(_fmt_reason
        "version ${fmt_VERSION} was found, but ${FMT_VERSION} or newer is required"
    )
endif()

if(_fmt_reason STREQUAL "")
    message(STATUS "Using system fmt (version ${fmt_VERSION})")
elseif(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    message(
        FATAL_ERROR
        "profiler-hub requires fmt: ${_fmt_reason}. Provide it on CMAKE_PREFIX_PATH, or configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it."
    )
else()
    message(STATUS "Fetching fmt ${FMT_VERSION}: ${_fmt_reason}")
    include(FetchContent)

    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG ${FMT_VERSION}
        GIT_SHALLOW TRUE
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
endif()

unset(_fmt_reason)
