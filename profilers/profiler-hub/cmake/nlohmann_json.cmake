# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

option(
    PROFILER_HUB_USE_SYSTEM_NLOHMANN_JSON
    "Use system-installed nlohmann_json if available"
    ON
)

set(NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "nlohmann_json version")

if(PROFILER_HUB_USE_SYSTEM_NLOHMANN_JSON)
    find_package(nlohmann_json ${NLOHMANN_JSON_VERSION} QUIET)
endif()

if(nlohmann_json_FOUND)
    message(
        STATUS
        "Using system nlohmann_json (version ${nlohmann_json_VERSION})"
    )
else()
    message(
        STATUS
        "System nlohmann_json not found, fetching version ${NLOHMANN_JSON_VERSION}"
    )
    include(FetchContent)

    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v${NLOHMANN_JSON_VERSION}
        GIT_SHALLOW TRUE
    )

    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(nlohmann_json)
endif()
