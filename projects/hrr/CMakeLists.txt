# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(hrr LANGUAGES CXX)

option(HRR_BUILD_PLAYBACK "Build HRR playback tools" ON)
option(HRR_BUILD_TESTS "Build HRR unit tests" OFF)

# Public HIP headers from this tree, searched ahead of the SDK's. A HIP API added
# to the tree is declared here before any SDK that CI installs has it, so
set(HRR_HIP_INCLUDE_DIRS "" CACHE STRING
    "Directories of in-tree public HIP headers to search before the SDK's (host C++ only)")
foreach(_hrr_hip_dir IN LISTS HRR_HIP_INCLUDE_DIRS)
  include_directories(BEFORE "$<$<COMPILE_LANGUAGE:CXX>:${_hrr_hip_dir}>")
endforeach()

# Shared HRR archive wire format (header-only public interface).
add_library(hrr_api_args INTERFACE)
add_library(hrr::api_args ALIAS hrr_api_args)

target_include_directories(hrr_api_args INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(DIRECTORY include/ DESTINATION include)

# The playback subdirectory defines both the CPU-only hrr_reader library and the
# HIP-dependent hrr-playback tool. hrr_reader is needed by the unit tests even
if((HRR_BUILD_PLAYBACK OR HRR_BUILD_TESTS) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/playback/CMakeLists.txt")
  add_subdirectory(playback)
endif()

if(HRR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
