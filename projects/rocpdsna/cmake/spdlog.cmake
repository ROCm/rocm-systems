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

option(ROCPDSNA_USE_SYSTEM_SPDLOG "Use system-installed spdlog if available" ON)

set(SPDLOG_VERSION "1.14.1" CACHE STRING "spdlog version")

if(ROCPDSNA_USE_SYSTEM_SPDLOG)
    find_package(spdlog ${SPDLOG_VERSION} QUIET)
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
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)

    # Spdlog workaround for building static library
    set(_ROCPDSNA_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    FetchContent_MakeAvailable(spdlog)

    set(BUILD_SHARED_LIBS ${_ROCPDSNA_BUILD_SHARED_LIBS_BACKUP})
    unset(_ROCPDSNA_BUILD_SHARED_LIBS_BACKUP)

    if(TARGET spdlog)
        set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    if(NOT TARGET spdlog::spdlog)
        add_library(spdlog::spdlog ALIAS spdlog)
    endif()
endif()
