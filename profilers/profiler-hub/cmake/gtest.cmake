# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(GTEST_VERSION "1.14.0" CACHE STRING "Minimum Google Test version")

# Fetching is Off by default: a missing or old package errors out.
if(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    find_package(GTest ${GTEST_VERSION})

    if(NOT GTest_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires GoogleTest ${GTEST_VERSION} or newer on CMAKE_PREFIX_PATH. Configure with -DPROFILER_HUB_BUILD_TESTS=OFF to skip the unit tests, or with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it instead."
        )
    endif()

    message(STATUS "Using system GoogleTest (version ${GTest_VERSION})")
else()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v${GTEST_VERSION}
        GIT_SHALLOW TRUE
        # Without this, the MakeAvailable below always fetches.
        # FetchContent derives the find_package call from the content name, which
        # here is not the name GoogleTest installs itself under.
        FIND_PACKAGE_ARGS ${GTEST_VERSION} NAMES GTest
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    # Tries find_package() first, fetches only if that fails.
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
