# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

if(ROCPROFSYS_BUILD_GTEST)
    message(STATUS "Setting up GoogleTest using FetchContent")

    include(FetchContent)

    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/googletest
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE CMakeLists.txt
        REPO_URL https://github.com/google/googletest.git
        REPO_BRANCH "v1.17.0"
    )

    # Configure GoogleTest options before adding it
    set(BUILD_GMOCK ON CACHE BOOL "Build GMock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest installation" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT" FORCE)

    # Declare GoogleTest from the submodule
    FetchContent_Declare(googletest SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/googletest)

    # Make GoogleTest available
    FetchContent_MakeAvailable(googletest)

    # Create interface library that wraps GoogleTest targets
    add_library(rocprofiler-systems-googletest-library INTERFACE)

    target_link_libraries(
        rocprofiler-systems-googletest-library
        INTERFACE GTest::gtest GTest::gtest_main GTest::gmock
    )

    message(STATUS "GoogleTest configured successfully using FetchContent")
else()
    message(STATUS "Using system GTest library")
    find_package(GTest REQUIRED)
    add_library(rocprofiler-systems-googletest-library INTERFACE)

    # Link against system GTest targets
    target_link_libraries(
        rocprofiler-systems-googletest-library
        INTERFACE GTest::gtest GTest::gtest_main
    )

    # Also link gmock if available
    if(TARGET GTest::gmock)
        target_link_libraries(
            rocprofiler-systems-googletest-library
            INTERFACE GTest::gmock
        )
    endif()
endif()
