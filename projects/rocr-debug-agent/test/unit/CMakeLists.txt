# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

#
# rocm-debug-agent unit tests have the following dependencies:
#
# GoogleTest (gtest)
# Threads
#
# Unit tests are pure logic tests with no GPU/HIP runtime dependencies.
# They can run on any machine without GPU hardware.

cmake_minimum_required(VERSION 3.21)

# GoogleTest resolution strategy with layered fallback.
# Step 1: Try CMake config package (TheRock CI, pre-installed systems).
find_package(GTest CONFIG QUIET)

# Step 2: Try CMake module (system installations).
if(NOT GTest_FOUND)
    find_package(GTest MODULE QUIET)
endif()

# If GoogleTest not found, fail the build.
if(NOT GTest_FOUND)
    message(FATAL_ERROR "GoogleTest not found. Unit tests are enabled but GoogleTest is not available. "
                        "Install GoogleTest or disable unit tests with -DENABLE_UNIT_TESTS=OFF")
endif()

# Find Threads (required by GTest and for pthreads).
find_package(Threads REQUIRED)

# Create unit test executable.
add_executable(
    rocm-debug-agent-unit-tests
    test_smoke.cpp
)

# Link libraries.
target_link_libraries(
    rocm-debug-agent-unit-tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

# C++ standard and compile options (match main library).
set_target_properties(
    rocm-debug-agent-unit-tests
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

target_compile_options(
    rocm-debug-agent-unit-tests
    PRIVATE
        -Werror -Wall -Wno-attributes
)

# Enable CTest integration.
include(GoogleTest)
gtest_discover_tests(rocm-debug-agent-unit-tests)

# Set INSTALL_RPATH for installed binary.
# Unit tests are installed at tests/rocm-debug-agent/, so to reach lib/
# we need $ORIGIN/../../lib (two levels up).
# Set THEROCK_NO_INSTALL_RPATH to disable TheRock's automatic RPATH processing,
# then manually set INSTALL_RPATH.
set_target_properties(
    rocm-debug-agent-unit-tests
    PROPERTIES
        THEROCK_NO_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN/../../lib"
)

# Install alongside existing integration tests.
install(
    TARGETS rocm-debug-agent-unit-tests
    RUNTIME DESTINATION tests/rocm-debug-agent
    COMPONENT tests
)
