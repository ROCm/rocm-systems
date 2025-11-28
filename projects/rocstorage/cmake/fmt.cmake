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

option(ROCSTORAGE_USE_SYSTEM_FMT "Use system-installed fmt if available" ON)

set(FMT_VERSION "10.2.1" CACHE STRING "fmt version")

if(ROCSTORAGE_USE_SYSTEM_FMT)
    find_package(fmt ${FMT_VERSION} QUIET)
endif()

if(fmt_FOUND)
    message(STATUS "Using system fmt (version ${fmt_VERSION})")
else()
    message(STATUS "System fmt not found, fetching version ${FMT_VERSION}")
    include(FetchContent)

    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG ${FMT_VERSION}
        GIT_SHALLOW TRUE
    )

    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    set(FMT_TEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(fmt)

    if(NOT TARGET fmt::fmt)
        add_library(fmt::fmt ALIAS fmt)
    endif()
endif()
