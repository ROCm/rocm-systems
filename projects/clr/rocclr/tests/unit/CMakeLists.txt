# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Host-side unit tests for ROCclr internals. Reached from the CLR build with
# -DBUILD_TESTS=ON (the same switch that enables the OpenCL tests; testing is
# enabled at the CLR root so `ctest` finds these from the top-level build
# directory), or configured standalone: cmake -S <this dir> -B build.
cmake_minimum_required(VERSION 3.16.8)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocclr_unit_tests LANGUAGES CXX)
  enable_testing()
endif()

# Prefer an installed GoogleTest; otherwise fetch the pinned release.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  # Match the parent project's CRT on Windows instead of gtest's default.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(rocclr_unit_tests memobjmap_erase_test.cpp)
target_compile_features(rocclr_unit_tests PRIVATE cxx_std_17)
target_include_directories(rocclr_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(rocclr_unit_tests PRIVATE GTest::gtest_main)

add_test(NAME rocclr_unit_tests COMMAND rocclr_unit_tests)
