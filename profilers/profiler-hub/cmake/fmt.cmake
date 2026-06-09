# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(FMT_VERSION "11.2.0" CACHE STRING "fmt version")

find_package(fmt ${FMT_VERSION} QUIET)

if(fmt_FOUND)
    message(STATUS "Using system fmt (version ${fmt_VERSION})")
else()
    message(STATUS "System fmt not found, fetching version ${FMT_VERSION}")
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

    FetchContent_MakeAvailable(fmt)

    if(TARGET fmt)
        set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    if(NOT TARGET fmt::fmt)
        add_library(fmt::fmt ALIAS fmt)
    endif()
endif()
