# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(FetchContent)

# We use find_package manually instead of FIND_PACKAGE_ARGS with
# FetchContent_Declare, since FIND_PACKAGE_ARGS was introduced in CMake
# version 3.24, which is unavailable on some operating systems:
# https://cmake.org/cmake/help/latest/module/FetchContent.html
find_package(ROCmCMakeBuildTools CONFIG QUIET)
if(NOT ROCmCMakeBuildTools_FOUND)
# lint_cmake: -readability/wonkycase
    FetchContent_Declare(
        rocm-cmake
        URL https://github.com/ROCm/rocm-cmake/archive/refs/tags/rocm-6.4.3.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP true
        SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
        SYSTEM
    )
    FetchContent_MakeAvailable(rocm-cmake)
# lint_cmake: +readability/wonkycase
endif()

if(rocm-cmake_SOURCE_DIR)
    message(STATUS "Using fetched rocm-cmake.")
    find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
    message(STATUS "Using system rocm-cmake.")
endif()
