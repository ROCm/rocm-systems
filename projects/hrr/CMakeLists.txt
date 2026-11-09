# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(hrr LANGUAGES CXX)

option(HRR_BUILD_PLAYBACK "Build HRR playback tools" ON)
option(HRR_BUILD_TESTS "Build HRR unit tests" OFF)

# Public HIP headers from this tree, searched ahead of the SDK's, so an API added here is declared before any CI SDK has it.
set(HRR_HIP_INCLUDE_DIRS "" CACHE STRING
    "Directories of in-tree public HIP headers to search before the SDK's (host C++ only)")
foreach(_hrr_hip_dir IN LISTS HRR_HIP_INCLUDE_DIRS)
  include_directories(BEFORE "$<$<COMPILE_LANGUAGE:CXX>:${_hrr_hip_dir}>")
endforeach()

add_library(hrr_api_args INTERFACE)
add_library(hrr::api_args ALIAS hrr_api_args)

target_include_directories(hrr_api_args INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(DIRECTORY include/ DESTINATION include)

# The playback subdirectory defines hrr_reader (CPU-only) and hrr-playback (HIP); the unit tests need hrr_reader even without HIP.
if((HRR_BUILD_PLAYBACK OR HRR_BUILD_TESTS) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/playback/CMakeLists.txt")
  add_subdirectory(playback)
endif()

if(HRR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
