# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
