# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(FetchContent)

# Use the rocm-cmake that comes with ROCm, or fetch it if there isn't
# one. FetchContent_Declare's FIND_PACKAGE_ARGS can't do this. The
# package has to be found as ROCmCMakeBuildTools, the name a
# superproject such as TheRock provides it under, so that has to be the
# content name too. When it's fetched, FetchContent_MakeAvailable()
# then writes a stub ROCmCMakeBuildToolsConfig.cmake to
# CMAKE_FIND_PACKAGE_REDIRECTS_DIR, which every later find_package()
# for it loads instead of the fetched one.
find_package(ROCmCMakeBuildTools CONFIG QUIET)
if(NOT ROCmCMakeBuildTools_FOUND)
# lint_cmake: -readability/wonkycase
    FetchContent_Declare(
        rocm-cmake
        GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
        GIT_TAG ecc716b97c2239cff00422ed7a43cd52a0839a0e  # rocm-6.4.3
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
