# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(SPDLOG_VERSION "1.15.3" CACHE STRING "Minimum spdlog version")

find_package(spdlog QUIET)

set(_spdlog_reason "")
if(NOT spdlog_FOUND)
    set(_spdlog_reason "no package was found on CMAKE_PREFIX_PATH")
elseif(spdlog_VERSION VERSION_LESS SPDLOG_VERSION)
    set(_spdlog_reason
        "version ${spdlog_VERSION} was found, but ${SPDLOG_VERSION} or newer is required"
    )
else()
    # A system spdlog is only safe to reuse if it was built with external fmt.
    # If it was built against its bundled fmt, its public headers pull in
    # <spdlog/fmt/bundled/...> and libspdlog exports bundled-fmt symbols, which
    # would coexist with the external fmt that profiler-hub links - two fmt
    # copies in one binary. Detect this via the interface compile definition
    # that spdlog's exported target carries when SPDLOG_FMT_EXTERNAL was set.
    get_target_property(
        _spdlog_iface_defs
        spdlog::spdlog
        INTERFACE_COMPILE_DEFINITIONS
    )
    if(NOT _spdlog_iface_defs MATCHES "SPDLOG_FMT_EXTERNAL")
        set(_spdlog_reason
            "version ${spdlog_VERSION} was found, but it is built against its bundled fmt, which would put two fmt copies in one binary"
        )
    endif()
    unset(_spdlog_iface_defs)
endif()

if(_spdlog_reason STREQUAL "")
    message(STATUS "Using system spdlog (version ${spdlog_VERSION})")
elseif(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    message(
        FATAL_ERROR
        "profiler-hub requires spdlog: ${_spdlog_reason}. Provide it on CMAKE_PREFIX_PATH, or configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it."
    )
else()
    message(STATUS "Fetching spdlog ${SPDLOG_VERSION}: ${_spdlog_reason}")
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

unset(_spdlog_reason)
