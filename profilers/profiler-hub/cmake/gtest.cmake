# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(GTEST_VERSION "1.14.0" CACHE STRING "Minimum Google Test version")

find_package(GTest QUIET)

set(_gtest_reason "")
if(NOT GTest_FOUND)
    set(_gtest_reason "no package was found on CMAKE_PREFIX_PATH")
elseif(GTest_VERSION STREQUAL "")
    # FindGTest reports no version when it resolves GoogleTest from bare
    # libraries rather than from an installed CMake package.
    set(_gtest_reason
        "a package was found, but it reports no version, so the ${GTEST_VERSION} floor cannot be checked"
    )
elseif(GTest_VERSION VERSION_LESS GTEST_VERSION)
    set(_gtest_reason
        "version ${GTest_VERSION} was found, but ${GTEST_VERSION} or newer is required"
    )
endif()

if(_gtest_reason STREQUAL "")
    message(STATUS "Using system GoogleTest (version ${GTest_VERSION})")
elseif(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    message(
        FATAL_ERROR
        "profiler-hub requires GoogleTest: ${_gtest_reason}. Provide it on CMAKE_PREFIX_PATH, configure with -DPROFILER_HUB_BUILD_TESTS=OFF, or configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it."
    )
else()
    message(STATUS "Fetching GoogleTest ${GTEST_VERSION}: ${_gtest_reason}")
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

unset(_gtest_reason)

include(GoogleTest)
