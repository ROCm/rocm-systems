################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and/or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

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

# If GoogleTest not found, fail the build.
if(NOT GTest_FOUND)
    message(FATAL_ERROR "GoogleTest not found. Install GoogleTest or disable with -DENABLE_UNIT_TESTS=OFF")
endif()

# Find amd-dbgapi (headers are needed for agent_utils.h includes).
find_package(
    amd-dbgapi
    REQUIRED
    CONFIG
    PATHS /opt/rocm/
    PATH_SUFFIXES cmake/amd-dbgapi lib/cmake/amd-dbgapi
)

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
        ${CMAKE_SOURCE_DIR}/src/agent_utils.cpp
        ${CMAKE_SOURCE_DIR}/src/logging.cpp
)

# Include directories.
target_include_directories(
    rocm-debug-agent-unit-tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
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

# Install (match integration test pattern).
install(
    TARGETS rocm-debug-agent-unit-tests
    RUNTIME DESTINATION tests/rocm-debug-agent/unit
    COMPONENT tests
)
