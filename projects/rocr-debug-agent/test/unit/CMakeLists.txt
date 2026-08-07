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
# Note: Test files will be added in subsequent commits.
# This commit only sets up the GoogleTest infrastructure.
