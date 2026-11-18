# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(SPDLOG_VERSION "1.15.3" CACHE STRING "Minimum spdlog version")

# Fetching is Off by default: a missing or old package errors out.
if(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    find_package(spdlog ${SPDLOG_VERSION})

    if(NOT spdlog_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires spdlog ${SPDLOG_VERSION} or newer on CMAKE_PREFIX_PATH. Configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it instead."
        )
    endif()

    message(STATUS "Using system spdlog (version ${spdlog_VERSION})")
else()
    include(FetchContent)

    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v${SPDLOG_VERSION}
        GIT_SHALLOW TRUE
        # Without this, the MakeAvailable below always fetches.
        FIND_PACKAGE_ARGS ${SPDLOG_VERSION}
    )

    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)

    # Spdlog workaround for building static library
    set(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    # Tries find_package() first, fetches only if that fails.
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

# The exported target's interface definitions are the only reliable record
# of which fmt a prebuilt spdlog was compiled against.
get_target_property(
    _spdlog_iface_defs
    spdlog::spdlog
    INTERFACE_COMPILE_DEFINITIONS
)

if(NOT _spdlog_iface_defs MATCHES "SPDLOG_FMT_EXTERNAL")
    message(
        FATAL_ERROR
        "profiler-hub requires an spdlog built with SPDLOG_FMT_EXTERNAL. The one provided is built against its bundled fmt, which would put two fmt copies in one binary. Provide a suitable spdlog on CMAKE_PREFIX_PATH, or configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON -DCMAKE_DISABLE_FIND_PACKAGE_spdlog=ON to bypass it and download one."
    )
endif()

unset(_spdlog_iface_defs)
