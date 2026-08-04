# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "nlohmann_json version")

# Always fetch rather than find_package(): some super-project build orchestrators
# (e.g. TheRock) intercept find_package() calls and reject any package not
# explicitly declared as a dependency of this subproject. Vendoring
# unconditionally avoids that dependency-declaration requirement entirely.
message(STATUS "Fetching nlohmann_json version ${NLOHMANN_JSON_VERSION}")
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
