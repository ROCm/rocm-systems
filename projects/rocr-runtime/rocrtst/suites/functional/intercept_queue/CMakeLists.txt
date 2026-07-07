##
## Copyright © Advanced Micro Devices, Inc., or its affiliates.
## SPDX-License-Identifier: MIT
##

## GPU-free unit tests for the InterceptQueue retry/overflow slot-accounting logic.
## These tests link only against GoogleTest and the pure-header decision logic in
## hsa-runtime's core/inc/intercept_queue_logic.h, so they run as an ordinary host
## executable with no GPU, libhsakmt, or the hsa-runtime library.
##
## Within the normal rocrtst build the test is compiled directly by
## suites/test_common/CMakeLists.txt (like the hotswap tests). This standalone
## CMakeLists.txt lets the fast GPU-free CI configure just this test:
##   cmake -S rocrtst/suites/functional/intercept_queue -B build
## which needs only a C++17 compiler and GoogleTest (or the bundled rocrtst/gtest).
cmake_minimum_required(VERSION 3.16)
project(intercept_queue_logic_unit_tests CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
enable_testing()

set(HSA_INTERCEPT_LOGIC_TEST intercept_queue_logic_test)

## intercept_queue dir -> hsa-runtime root (headers are included as "core/inc/...").
get_filename_component(_hsa_runtime_root
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../runtime/hsa-runtime ABSOLUTE)

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
  get_filename_component(_bundled_gtest ${CMAKE_CURRENT_SOURCE_DIR}/../../../gtest ABSOLUTE)
  if(EXISTS ${_bundled_gtest}/src/gtest.cpp)
    ## Bundled GoogleTest is compiled from its individual sources. gtest_main.cpp is
    ## intentionally excluded; we provide our own main.cc.
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
