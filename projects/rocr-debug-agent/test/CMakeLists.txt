################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
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
# rocm-debug-agent tests have the following dependencies:
#
# rocr-debug-agent
# HIP compiler/tools
#
#
# Available defines to be used when building rocr-debug-agent-test:
#
# GPU_TARGETS (optional)
#
# Build tests from sources for a list of GPU targets.
#
# GPU_TARGETS lists one or more GPU targets. If set, build the test files
# for each entry in GPU target. If not set, fallback to building for
# whatever GPU target the tools can find on the system.
#
#
# ENABLE_TESTS
#
# Enable the rocr-debug-agent-test testing target. Default is OFF.
#
# As part of building the rocr-debug-agent tests, we also add a testing
# target that can be invoked to execute the tests. Alternatively the tests
# can be invoked by hand.

cmake_minimum_required(VERSION 3.21)

project(rocm-debug-agent-test LANGUAGES CXX HIP)

include(GNUInstallDirs)

# By default do not include the testing target.
option(ENABLE_TESTS "Enable building Debug Agent testing target" OFF)

set(Python3_FIND_VIRTUALENV "FIRST")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Try CONFIG mode first (TheRock), then fall back to discovery (standalone).
find_package(hip CONFIG QUIET)
if(NOT hip_FOUND)
    # Auto-discover HIP cmake files in standard ROCm installation paths
    find_package(
        hip
        REQUIRED
        CONFIG
        PATHS /opt/rocm
        PATH_SUFFIXES cmake/hip lib/cmake/hip hip/cmake
    )
endif()

set(MAIN_SOURCES
  debug_agent_test.cpp
  vector_add_assert_trap.cpp
  sigquit.cpp
  print_all_waves.cpp
  snapshot_objfile_on_load.cpp
  vector_add_normal.cpp
  vector_add_memory_fault.cpp
  save_code_objects.cpp)
set(NO_DEBUG_SOURCES vector_add_assert_trap_no_debug_info.cpp)
set(HEADERS util.h)

# Source files required for testing.
set(TEST_SOURCE_FILES
    vector_add_assert_trap.cpp
)

# Mark source files as HIP.
set_source_files_properties(${MAIN_SOURCES} ${NO_DEBUG_SOURCES} PROPERTIES LANGUAGE HIP)

# Build offload args ONLY for .hipfb custom commands.
if(CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "OFF")
  set(OFFLOAD_ARGS "")
  foreach(ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
    list(APPEND OFFLOAD_ARGS "--offload-arch=${ARCH}")
  endforeach()
  message(STATUS "Building for HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
  # CMake's HIP support should take care of picking up a fallback HIP
  # architecture in case no GPU is found on the system. The default is
  # usually gfx906.
  message(FATAL_ERROR "Could not determine HIP architecture for building device code."
            "Please set CMAKE_HIP_ARCHITECTURES to the desired architecture.")
endif()

# Set HIP_FLAGS for compilation of hipfb files.
#
# "-x hip" because we are using clang++ instead of hipcc.
#
# OFFLOAD_ARGS for the offloading architectures.
#
# "-O3" because test 3 of debug agent requires it.
#
# -cuda-device-only to generate hipfb files.
#
set(HIP_FLAGS -x hip ${OFFLOAD_ARGS} -O3 --cuda-device-only)

# This is needed because some debug agent tests rely on looking up the source
# files through DWARF entries. Since the test files and executable may be
# installed elsewhere and the original source files may not be available
# anymore, we have to adjust the source file path in DWARF to be the same
# directory of the test executable.
set(DWARF_FLAGS
  -fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.
  -fdebug-compilation-dir=.
)

# Output directory for the HIPFB files.
set(HIPFB_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Generate the hipfb files we use for testing.
add_custom_command(
  OUTPUT "${HIPFB_OUTPUT_DIR}/snapshot_objfile_on_load.hipfb"
  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_FLAGS}
          ${CMAKE_CURRENT_SOURCE_DIR}/snapshot_objfile_on_load.cpp
          -o ${HIPFB_OUTPUT_DIR}/snapshot_objfile_on_load.hipfb
  DEPENDS snapshot_objfile_on_load.cpp
  COMMENT "Building snapshot_objfile_on_load.hipfb"
  VERBATIM)

add_custom_command(
  OUTPUT "${HIPFB_OUTPUT_DIR}/save_code_objects.hipfb"
  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_FLAGS}
          ${CMAKE_CURRENT_SOURCE_DIR}/save_code_objects.cpp
          -o ${HIPFB_OUTPUT_DIR}/save_code_objects.hipfb
  DEPENDS save_code_objects.cpp
  COMMENT "Building save_code_objects.hipfb"
  VERBATIM)

# Define targets to manage dependencies.
add_custom_target(snapshot_objfile_on_load-tgt ALL
  DEPENDS ${HIPFB_OUTPUT_DIR}/snapshot_objfile_on_load.hipfb)
add_custom_target(save_code_objects-tgt ALL
  DEPENDS ${HIPFB_OUTPUT_DIR}/save_code_objects.hipfb)

# Build the library with no debug info.
add_library(rocm-debug-agent-no-dbginfo ${NO_DEBUG_SOURCES})
target_compile_options(rocm-debug-agent-no-dbginfo PRIVATE)

# Build the test executable with debug info.
add_executable(rocm-debug-agent-test ${MAIN_SOURCES})
target_compile_options(rocm-debug-agent-test PRIVATE -ggdb ${DWARF_FLAGS})
target_link_libraries(rocm-debug-agent-test PRIVATE rocm-debug-agent-no-dbginfo)
add_dependencies(rocm-debug-agent-test snapshot_objfile_on_load-tgt save_code_objects-tgt)

# Enable the testing target if requested. This enables running debug agent
# tests from the build tree with ctest.
#
# Tests can also be executed using the prebuilt test executable that is
# installed alongside the debug agent library.
if(ENABLE_TESTS)
    # In order to keep our ability to run tests within the build tree, we need
    # to copy the required source files over for tests to run properly.
    foreach(source_file IN LISTS TEST_SOURCE_FILES)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}"
            "${CMAKE_CURRENT_BINARY_DIR}/${source_file}"
            COPYONLY
        )
    endforeach()

    enable_testing()
    add_test(
        NAME rocm-debug-agent-test
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run-test.py
            ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# Install the test executable.
install(
    TARGETS rocm-debug-agent-test
    DESTINATION tests/rocm-debug-agent COMPONENT tests
)

# Install the shared objects used by testing.
install(
    FILES
        ${HIPFB_OUTPUT_DIR}/snapshot_objfile_on_load.hipfb
        ${HIPFB_OUTPUT_DIR}/save_code_objects.hipfb
    DESTINATION tests/rocm-debug-agent
    COMPONENT tests
)

# Install the required source files and the testing script.
install(
    FILES run-test.py ${TEST_SOURCE_FILES}
    DESTINATION tests/rocm-debug-agent
    COMPONENT tests
)
