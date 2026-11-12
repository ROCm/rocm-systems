# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(PROFILER_HUB_ZLIB_VERSION "1.3.2" CACHE STRING "Minimum zlib version")

# Fetching is Off by default: a missing or old package errors out.
if(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    find_package(ZLIB ${PROFILER_HUB_ZLIB_VERSION})

    if(NOT ZLIB_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires zlib ${PROFILER_HUB_ZLIB_VERSION} or newer on CMAKE_PREFIX_PATH. Configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it instead, or with -DPROFILER_HUB_ENABLE_PYTORCH_KINETO=OFF to leave PyTorch traces out."
        )
    endif()

    message(STATUS "Using system zlib (version ${ZLIB_VERSION})")
else()
    include(FetchContent)

    FetchContent_Declare(
        ZLIB
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v${PROFILER_HUB_ZLIB_VERSION}
        GIT_SHALLOW TRUE
        # Without this, the MakeAvailable below always fetches.
        FIND_PACKAGE_ARGS ${PROFILER_HUB_ZLIB_VERSION}
    )

    # zlib ignores BUILD_SHARED_LIBS: it builds a shared and a static library
    # unless told otherwise.
    set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)

    # Force a static, PIC zlib regardless of a parent project's BUILD_SHARED_LIBS
    # (e.g. rocprofiler-systems sets it ON project-wide, which would otherwise
    # produce a libz.so that ZLIB_INSTALL=OFF never installs, leaving it
    # unresolvable at runtime).
    set(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    # Tries find_package() first, fetches only if that fails.
    FetchContent_MakeAvailable(ZLIB)

    set(BUILD_SHARED_LIBS ${_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP})
    unset(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP)

    if(TARGET zlibstatic)
        set_target_properties(
            zlibstatic
            PROPERTIES POSITION_INDEPENDENT_CODE ON
        )
    endif()

    if(NOT TARGET ZLIB::ZLIB)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()
endif()
