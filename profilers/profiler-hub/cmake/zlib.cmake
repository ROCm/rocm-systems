# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

# The loader only needs inflateInit2, inflate, inflateReset and inflateEnd, which
# every zlib 1.2.x has, so a distribution's package is accepted from this
# version on. The version that is fetched when none is found is pinned apart.
set(ZLIB_MIN_VERSION "1.2.11" CACHE STRING "Minimum zlib version")
set(ZLIB_FETCH_VERSION "1.3.1" CACHE STRING "zlib version to download")

# FetchContent's own find_package() step looks for a package config file, which
# zlib packages do not ship, so CMake's find module is called here instead.
# Fetching forced on by FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER skips it, as it
# does for the other dependencies.
if(
    NOT PROFILER_HUB_FETCH_DEPENDENCIES
    OR NOT FETCHCONTENT_TRY_FIND_PACKAGE_MODE STREQUAL "NEVER"
)
    find_package(ZLIB ${ZLIB_MIN_VERSION})
endif()

if(ZLIB_FOUND)
    message(STATUS "Using system zlib (version ${ZLIB_VERSION_STRING})")
elseif(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    # Fetching is Off by default: a missing or old package errors out.
    message(
        FATAL_ERROR
        "profiler-hub requires zlib ${ZLIB_MIN_VERSION} or newer on CMAKE_PREFIX_PATH to read PyTorch traces. Configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it instead, or with -DPROFILER_HUB_ENABLE_PYTORCH_KINETO=OFF to leave PyTorch traces out."
    )
else()
    include(FetchContent)

    FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v${ZLIB_FETCH_VERSION}
        GIT_SHALLOW TRUE
    )

    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SKIP_INSTALL_ALL ON)

    FetchContent_MakeAvailable(zlib)

    unset(SKIP_INSTALL_ALL)

    set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(ZLIB::ZLIB ALIAS zlibstatic)
endif()
