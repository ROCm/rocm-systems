# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "Minimum nlohmann_json version")

# Fetching is Off by default: a missing or old package errors out.
if(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    find_package(nlohmann_json ${NLOHMANN_JSON_VERSION})

    if(NOT nlohmann_json_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires nlohmann_json ${NLOHMANN_JSON_VERSION} or newer on CMAKE_PREFIX_PATH. Configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it instead."
        )
    endif()

    message(
        STATUS
        "Using system nlohmann_json (version ${nlohmann_json_VERSION})"
    )
else()
    include(FetchContent)

    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v${NLOHMANN_JSON_VERSION}
        GIT_SHALLOW TRUE
        # Without this, the MakeAvailable below always fetches.
        FIND_PACKAGE_ARGS ${NLOHMANN_JSON_VERSION}
    )

    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)

    # Tries find_package() first, fetches only if that fails.
    FetchContent_MakeAvailable(nlohmann_json)
endif()
