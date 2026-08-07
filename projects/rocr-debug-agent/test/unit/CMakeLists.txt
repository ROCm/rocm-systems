# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

#
# rocm-debug-agent unit tests have the following dependencies:
#
# GoogleTest (gtest)
# amd-dbgapi (headers only - for type declarations in agent_utils.h)
# Threads
#
# Unit tests are pure logic tests with no GPU/HIP/dbgapi runtime dependencies.
# They can run on any machine without GPU hardware.
#
# Note: agent_utils.h includes amd-dbgapi headers for type declarations
# (amd_dbgapi_exceptions_t, amd_dbgapi_wave_stop_reasons_t). This is a
# compile-time only dependency - no dbgapi functions are called at runtime.

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

# Find amd-dbgapi for headers only (compile-time dependency).
# We need the header for type declarations but do NOT link the library.
find_package(amd-dbgapi REQUIRED CONFIG)

# Find Threads (required by GTest and for pthreads).
find_package(Threads REQUIRED)

# Create unit test executable.
add_executable(
    rocm-debug-agent-unit-tests
    test_uri.cpp
    test_formatting.cpp
)

# Link directly to source files (not the .so).
target_sources(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/agent_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/logging.cpp
)

# Include directories.
# Use generator expression to get amd-dbgapi include dir without linking.
target_include_directories(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
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

# Install (match integration test pattern).
install(
    TARGETS rocm-debug-agent-unit-tests
    RUNTIME DESTINATION tests/rocm-debug-agent/unit
    COMPONENT tests
)
