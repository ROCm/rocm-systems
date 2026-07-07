##
## Copyright © Advanced Micro Devices, Inc., or its affiliates.
## SPDX-License-Identifier: MIT
##

## GPU-free unit tests for the InterceptQueue retry/overflow slot-accounting logic.
## These tests link only against GoogleTest and the pure-header decision logic in
## core/inc/intercept_queue_logic.h, so they run as an ordinary host executable.

## This directory can be configured two ways:
##   * via add_subdirectory() from the hsa-runtime build (when HSA_BUILD_TESTS=ON), or
##   * standalone (e.g. from CI: cmake -S core/runtime/test -B build), which needs no GPU,
##     libhsakmt, or the hsa-runtime library -- only a C++17 compiler and GoogleTest.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(hsa_runtime_unit_tests CXX)
  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()
  enable_testing()
endif()

set(HSA_INTERCEPT_LOGIC_TEST intercept_queue_logic_test)

## core/runtime/test -> hsa-runtime root (headers are included as "core/inc/...").
get_filename_component(_hsa_runtime_root ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)

find_package(Threads REQUIRED)

add_executable(${HSA_INTERCEPT_LOGIC_TEST}
  main.cc
  intercept_queue_logic_test.cpp)

target_include_directories(${HSA_INTERCEPT_LOGIC_TEST} PRIVATE ${_hsa_runtime_root})

## Resolve GoogleTest: prefer an installed/config package, otherwise fall back to the
## GoogleTest sources bundled with rocrtst so the test also builds without network access.
find_package(GTest QUIET)
if(TARGET GTest::gtest)
  target_link_libraries(${HSA_INTERCEPT_LOGIC_TEST} PRIVATE GTest::gtest Threads::Threads)
else()
  get_filename_component(_bundled_gtest ${_hsa_runtime_root}/../../rocrtst/gtest ABSOLUTE)
  if(EXISTS ${_bundled_gtest}/src/gtest.cpp)
    ## Bundled GoogleTest is compiled from its individual sources (gtest-all.cpp is a stub
    ## here). gtest_main.cpp is intentionally excluded; we provide our own main.cc.
    target_sources(${HSA_INTERCEPT_LOGIC_TEST} PRIVATE
      ${_bundled_gtest}/src/gtest.cpp
      ${_bundled_gtest}/src/gtest-port.cpp
      ${_bundled_gtest}/src/gtest-printers.cpp
      ${_bundled_gtest}/src/gtest-filepath.cpp
      ${_bundled_gtest}/src/gtest-test-part.cpp
      ${_bundled_gtest}/src/gtest-typed-test.cpp
      ${_bundled_gtest}/src/gtest-death-test.cpp)
    target_include_directories(${HSA_INTERCEPT_LOGIC_TEST} PRIVATE
      ${_bundled_gtest}/include
      ${_bundled_gtest})
    target_link_libraries(${HSA_INTERCEPT_LOGIC_TEST} PRIVATE Threads::Threads)
  else()
    message(FATAL_ERROR
      "GoogleTest not found: install GTest or provide the bundled rocrtst/gtest sources "
      "to build ${HSA_INTERCEPT_LOGIC_TEST}.")
  endif()
endif()

add_test(NAME ${HSA_INTERCEPT_LOGIC_TEST} COMMAND ${HSA_INTERCEPT_LOGIC_TEST})
