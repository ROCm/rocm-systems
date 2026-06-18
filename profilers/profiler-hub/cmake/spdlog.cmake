# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(SPDLOG_VERSION "1.15.3" CACHE STRING "spdlog version")

find_package(spdlog ${SPDLOG_VERSION} QUIET)

# A system spdlog is only safe to reuse if it was built with external fmt.
# If it was built against its bundled fmt, its public headers pull in
# <spdlog/fmt/bundled/...> and libspdlog exports bundled-fmt symbols, which
# would coexist with the external fmt that profiler-hub links - two fmt
# copies in one binary. Detect this via the interface compile definition
# that spdlog's exported target carries when SPDLOG_FMT_EXTERNAL was set.
if(spdlog_FOUND)
    get_target_property(
        _spdlog_iface_defs
        spdlog::spdlog
        INTERFACE_COMPILE_DEFINITIONS
    )
    if(NOT _spdlog_iface_defs MATCHES "SPDLOG_FMT_EXTERNAL")
        message(
            STATUS
            "System spdlog uses bundled fmt; falling back to FetchContent with external fmt"
        )
        set(spdlog_FOUND FALSE)
    endif()
endif()

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
