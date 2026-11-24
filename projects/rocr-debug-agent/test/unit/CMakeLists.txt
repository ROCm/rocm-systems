# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

#
# rocm-debug-agent unit tests have the following dependencies:
#
# GoogleTest (gtest)
# Threads
# amd-dbgapi (headers only)
#
# Unit tests are pure logic tests with no GPU/HIP runtime dependencies.
# They can run on any machine without GPU hardware.
#
# The modules under test are linked from the rocm-debug-agent-core object
# library, so the tests exercise the same objects as the shipping library.
# Only the amd-dbgapi headers are used; libamd-dbgapi is not linked.

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

# The rocm-debug-agent-core target is created by the top-level CMakeLists.txt.
if(NOT TARGET rocm-debug-agent-core)
    message(
        FATAL_ERROR
        "The rocm-debug-agent-core target is missing. "
        "Configure from the top-level rocr-debug-agent CMakeLists.txt."
    )
endif()

# Headers under test.
set(AGENT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

# Create unit test executable.
add_executable(
    rocm-debug-agent-unit-tests
    test_agent_config.cpp
    test_agent_dbgapi_utils.cpp
    test_agent_utils.cpp
    test_logging.cpp
)

# src/ for the headers under test.  The amd-dbgapi headers come from
# rocm-debug-agent-core.
target_include_directories(rocm-debug-agent-unit-tests PRIVATE ${AGENT_SRC_DIR})

# The rocm-debug-agent library exports only OnLoad and OnUnload (see
# src/exportmap), so the modules under test are linked from the same objects
# the library is built from, with the library's compile flags.
target_link_libraries(
    rocm-debug-agent-unit-tests
    PRIVATE
        rocm-debug-agent-core
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

# Same C++ standard and warning options as the library.
set_target_properties(
    rocm-debug-agent-unit-tests
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

target_compile_options(
    rocm-debug-agent-unit-tests
    PRIVATE ${AGENT_WARNING_OPTIONS}
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
