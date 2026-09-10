# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

#
# rocm-debug-agent unit tests have the following dependencies:
#
# GoogleTest (gtest)
# amd-dbgapi (headers only for type definitions)
# Threads
#
# Unit tests are pure logic tests with no GPU/HIP runtime dependencies.
# They can run on any machine without GPU hardware.
# The log_level.h header provides logging enum without requiring
# the logging implementation or amd-dbgapi library linkage.

cmake_minimum_required(VERSION 3.21)

# GoogleTest resolution strategy with layered fallback.
# Step 1: Try CMake config package (TheRock CI, pre-installed systems).
find_package(GTest CONFIG QUIET)

# Step 2: Try CMake module (system installations).
if(NOT GTest_FOUND)
    find_package(GTest MODULE QUIET)
endif()

# Step 3: Try manual library search.
if(NOT GTest_FOUND)
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
    find_library(GTEST_LIBRARY gtest)
    find_library(GTEST_MAIN_LIBRARY gtest_main)

    if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY AND GTEST_MAIN_LIBRARY)
        add_library(GTest::gtest UNKNOWN IMPORTED)
        set_target_properties(
            GTest::gtest
            PROPERTIES
                IMPORTED_LOCATION "${GTEST_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
        )

        add_library(GTest::gtest_main UNKNOWN IMPORTED)
        set_target_properties(
            GTest::gtest_main
            PROPERTIES
                IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
        )

        set(GTest_FOUND TRUE)
        message(STATUS "Found GoogleTest (manual): ${GTEST_LIBRARY}")
    endif()
endif()

# If GoogleTest not found, disable unit tests.
if(NOT GTest_FOUND)
    message(WARNING "GoogleTest not found. Unit tests will be disabled. "
                    "Install GoogleTest or disable with -DENABLE_UNIT_TESTS=OFF")
    return()
endif()

# Find amd-dbgapi headers for type definitions.
#
# TheRock: CMAKE_PREFIX_PATH automatically includes build/dist/rocm
# Standalone: Set CMAKE_PREFIX_PATH=/opt/rocm or custom install prefix
#
# Unit tests only need amd-dbgapi headers (not the library) because:
# - agent_utils.cpp only uses log_level_t enum (now in log_level.h)
# - logging.cpp is excluded from unit test build
# - No calls to amd_dbgapi_* functions in unit-tested code

find_package(
    amd-dbgapi
    REQUIRED
    CONFIG
)

message(STATUS "Found amd-dbgapi headers: ${amd-dbgapi_DIR}")

# Create unit test executable.
add_executable(
    rocm-debug-agent-unit-tests
    test_formatting.cpp
    test_uri.cpp
    test_options.cpp
    test_exceptions.cpp
    test_logging_stub.cpp
)

# Link directly to source files (not the .so).
# Note: logging.cpp excluded - test_logging_stub.cpp provides no-op implementations.
target_sources(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/../src/agent_utils.cpp
)

# Include directories.
target_include_directories(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/../src
)

# Add amd-dbgapi include directories (headers only, no library linking).
if(TARGET amd-dbgapi)
    get_target_property(
        DBGAPI_INCLUDE_DIRS
        amd-dbgapi
        INTERFACE_INCLUDE_DIRECTORIES
    )
    target_include_directories(
        rocm-debug-agent-unit-tests
        PRIVATE
            ${DBGAPI_INCLUDE_DIRS}
    )
endif()

# Link libraries.
# Note: amd-dbgapi library excluded - unit tests only need headers for type definitions.
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
