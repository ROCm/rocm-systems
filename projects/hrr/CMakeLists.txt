# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(hrr LANGUAGES CXX)

option(HRR_BUILD_PLAYBACK "Build HRR playback tools" ON)
option(HRR_BUILD_TESTS "Build HRR unit tests" OFF)

set(HRR_CLR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../clr" CACHE PATH
  "Path to the CLR source tree used by HRR code generation")
set(HRR_HIP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hip" CACHE PATH
  "Path to the HIP source tree used by HRR code generation")

include("${HRR_CLR_SOURCE_DIR}/cmake/HrrCodegen.cmake")
set(HRR_CODEGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
hrr_add_codegen(
  TARGET hrr_playback_codegen
  HRR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  CLR_SOURCE_DIR "${HRR_CLR_SOURCE_DIR}"
  HIP_SOURCE_DIR "${HRR_HIP_SOURCE_DIR}"
  OUTPUT_DIR "${HRR_CODEGEN_DIR}"
  PLAYBACK
)

# Shared HRR archive wire format (header-only public interface).
add_library(hrr_api_args INTERFACE)
add_library(hrr::api_args ALIAS hrr_api_args)

target_include_directories(hrr_api_args INTERFACE
  $<BUILD_INTERFACE:${HRR_CODEGEN_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_dependencies(hrr_api_args hrr_playback_codegen)

install(FILES
  "${HRR_CODEGEN_DIR}/include/hrr/hrr_api_args.h"
  "${HRR_CODEGEN_DIR}/include/hrr/hrr_regions.h"
  DESTINATION include/hrr
)

# The playback subdirectory defines both the CPU-only hrr_reader library and the
# HIP-dependent hrr-playback tool. hrr_reader is needed by the unit tests even
# when the playback tool itself is not built, so add the subdirectory when either
# playback or tests are enabled; hrr-playback and its HIP dependency are gated
# inside playback/CMakeLists.txt on HRR_BUILD_PLAYBACK.
if((HRR_BUILD_PLAYBACK OR HRR_BUILD_TESTS) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/playback/CMakeLists.txt")
  add_subdirectory(playback)
endif()

if(HRR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
