# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(SPDLOG_VERSION "1.14.1" CACHE STRING "spdlog version")

find_package(spdlog QUIET)

if(spdlog_FOUND)
    message(STATUS "Using system spdlog (version ${spdlog_VERSION})")
else()
    message(
        STATUS
        "System spdlog not found, fetching version ${SPDLOG_VERSION}"
    )
    include(FetchContent)

    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v${SPDLOG_VERSION}
        GIT_SHALLOW TRUE
    )

    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)

    # Spdlog workaround for building static library
    set(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    FetchContent_MakeAvailable(spdlog)

    set(BUILD_SHARED_LIBS ${_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP})
    unset(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP)

    if(TARGET spdlog)
        set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    if(NOT TARGET spdlog::spdlog)
        add_library(spdlog::spdlog ALIAS spdlog)
    endif()
endif()
