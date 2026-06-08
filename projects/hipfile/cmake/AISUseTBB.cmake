# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISSanitizers)
include(FetchContent)

# When building with the sanitizers, we HAVE to build TBB from source so that
# it picks up the sanitizer flags too.
if(AIS_USE_SANITIZERS OR AIS_USE_THREAD_SANITIZER)
    set(TBB_FOUND FALSE)
else()
    find_package(TBB QUIET)
endif()

if(NOT TBB_FOUND)
# lint_cmake: -readability/wonkycase
    FetchContent_Declare(
        TBB
        URL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2023.0.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP true
        SYSTEM
    )
# lint_cmake: +readability/wonkycase

    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    set(TBB_INSTALL OFF CACHE BOOL "" FORCE)
# lint_cmake: -readability/wonkycase
    FetchContent_MakeAvailable(TBB)
# lint_cmake: +readability/wonkycase
    message(STATUS "Using fetched TBB")
else()
    message(STATUS "Using system TBB")
endif()
