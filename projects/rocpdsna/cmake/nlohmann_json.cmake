# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include_guard(DIRECTORY)

option(
    ROCPDSNA_USE_SYSTEM_NLOHMANN_JSON
    "Use system-installed nlohmann_json if available"
    ON
)

set(NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "nlohmann_json version")

if(ROCPDSNA_USE_SYSTEM_NLOHMANN_JSON)
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
