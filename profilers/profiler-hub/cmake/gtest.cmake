# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

option(
    PROFILER_HUB_USE_SYSTEM_GTEST
    "Use system-installed GoogleTest if available"
    ON
)

set(GTEST_VERSION "1.14.0" CACHE STRING "Google Test version")

if(PROFILER_HUB_USE_SYSTEM_GTEST)
    find_package(GTest ${GTEST_VERSION} QUIET)
endif()

if(GTest_FOUND)
    message(STATUS "Using system GoogleTest (version ${GTest_VERSION})")
else()
    message(
        STATUS
        "System GoogleTest not found, fetching version ${GTEST_VERSION}"
    )
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v${GTEST_VERSION}
        GIT_SHALLOW TRUE
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()

    if(NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()

    if(NOT TARGET GTest::gmock)
        add_library(GTest::gmock ALIAS gmock)
    endif()

    if(NOT TARGET GTest::gmock_main)
        add_library(GTest::gmock_main ALIAS gmock_main)
    endif()
endif()

include(GoogleTest)
