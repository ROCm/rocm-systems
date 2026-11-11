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
# The sources under test are compiled directly into the test executable.  Only
# the amd-dbgapi headers are used; libamd-dbgapi is not linked.

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

# The amd-dbgapi target is created by the top-level CMakeLists.txt.
if(NOT TARGET amd-dbgapi)
    message(
        FATAL_ERROR
        "The amd-dbgapi target is missing. "
        "Configure from the top-level rocr-debug-agent CMakeLists.txt."
    )
endif()

# Sources under test.
set(AGENT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

# Create unit test executable.
add_executable(
    rocm-debug-agent-unit-tests
    test_agent_config.cpp
    test_agent_dbgapi_utils.cpp
    test_agent_utils.cpp
    test_logging.cpp
    # The rocm-debug-agent library exports only OnLoad and OnUnload (see
    # src/exportmap), so the sources under test are compiled directly.  The
    # library's compile definitions are not needed by these modules.
    ${AGENT_SRC_DIR}/agent_config.cpp
    ${AGENT_SRC_DIR}/agent_dbgapi_utils.cpp
    ${AGENT_SRC_DIR}/agent_utils.cpp
    ${AGENT_SRC_DIR}/logging.cpp
)

# src/ for the headers under test, amd-dbgapi for the enum types and version
# macros.
target_include_directories(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${AGENT_SRC_DIR}
        $<TARGET_PROPERTY:amd-dbgapi,INTERFACE_INCLUDE_DIRECTORIES>
)

# Link libraries.
target_link_libraries(
    rocm-debug-agent-unit-tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

# C++ standard and warning options (match main library).
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
