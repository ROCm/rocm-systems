# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

#
# rocm-debug-agent unit tests have the following dependencies:
#
# GoogleTest (gtest)
# amd-dbgapi (headers only, via pkg-config)
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

# If GoogleTest not found, disable unit tests.
if(NOT GTest_FOUND)
    message(WARNING "GoogleTest not found. Unit tests will be disabled. "
                    "Install GoogleTest or disable with -DENABLE_UNIT_TESTS=OFF")
    return()
endif()

# Find amd-dbgapi library.
#
# TheRock: CMAKE_PREFIX_PATH automatically includes build/dist/rocm
# Standalone: Set CMAKE_PREFIX_PATH=/opt/rocm or custom install prefix
#
# Unit tests need to link amd-dbgapi because:
# - agent_utils.cpp calls logging functions (log_level, detail::log)
# - logging.cpp calls amd_dbgapi_set_log_level()
# Therefore headers-only dependency is insufficient.

find_package(
    amd-dbgapi
    REQUIRED
    CONFIG
)

message(STATUS "Found amd-dbgapi: ${amd-dbgapi_DIR}")

# Create unit test executable.
add_executable(
    rocm-debug-agent-unit-tests
    test_formatting.cpp
    test_uri.cpp
    test_options.cpp
    test_exceptions.cpp
)

# Link directly to source files (not the .so).
target_sources(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/../src/agent_config.cpp
        ${PROJECT_SOURCE_DIR}/../src/agent_utils.cpp
        ${PROJECT_SOURCE_DIR}/../src/logging.cpp
)

# Include directories.
target_include_directories(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/../src
)

# Link libraries.
target_link_libraries(
    rocm-debug-agent-unit-tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        amd-dbgapi
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

# Define _GNU_SOURCE for getopt_long() support on Linux.
target_compile_definitions(
    rocm-debug-agent-unit-tests
    PRIVATE
        _GNU_SOURCE
        __STDC_LIMIT_MACROS
        __STDC_CONSTANT_MACROS
)

# Enable CTest integration.
include(GoogleTest)
gtest_discover_tests(rocm-debug-agent-unit-tests)

# Set INSTALL_RPATH for installed binary.
# Unit tests are installed at tests/rocm-debug-agent/unit, so to reach lib/
# we need $ORIGIN/../../../lib (three levels up).
# Set THEROCK_NO_INSTALL_RPATH to disable TheRock's automatic RPATH processing,
# then manually set INSTALL_RPATH.
set_target_properties(
    rocm-debug-agent-unit-tests
    PROPERTIES
        THEROCK_NO_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN/../../../lib"
)

# Install (match integration test pattern).
install(
    TARGETS rocm-debug-agent-unit-tests
    RUNTIME DESTINATION tests/rocm-debug-agent/unit
    COMPONENT tests
)
