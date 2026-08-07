# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

#
# rocm-debug-agent unit tests have the following dependencies:
#
# GoogleTest (gtest)
# amd-dbgapi
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

# Find amd-dbgapi (headers are needed for agent_utils.h includes).
find_package(
    amd-dbgapi
    REQUIRED
    CONFIG
)

# Create unit test executable.
# Note: Test files will be added in subsequent commits.
# This commit only sets up the GoogleTest infrastructure.
