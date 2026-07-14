# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

## Unit tests for device/ hostcall logic. Opt-in via -DCLR_BUILD_TESTS=ON.
## The tests are self-contained control-flow models (they do not link rocclr),
## so this directory can be built without the full runtime.

## gtest_discover_tests(... DISCOVERY_MODE PRE_TEST) requires CMake >= 3.18.
## The user opted in via CLR_BUILD_TESTS, so fail loudly rather than silently
## producing no tests.
if (CMAKE_VERSION VERSION_LESS 3.18)
  message(FATAL_ERROR
    "CLR_BUILD_TESTS requires CMake >= 3.18 "
    "(gtest_discover_tests DISCOVERY_MODE PRE_TEST); found ${CMAKE_VERSION}.")
endif()

## Threads::Threads is a transitive dependency of GTest::gtest_main.
find_package(Threads REQUIRED)

## Locate GTest, or fetch it if the build image doesn't provide one. Because the
## tests are opt-in, a missing GTest must NOT silently skip them -- fall back to
## FetchContent, mirroring projects/rocr-runtime core/util.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "clr: GTest not found on system -- fetching via FetchContent")
  include(FetchContent)
  set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest installation" FORCE)
  if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT for GTest" FORCE)
  endif()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_executable(devhostcall_test devhostcall_test.cpp)

target_link_libraries(devhostcall_test PRIVATE
  GTest::gtest
  GTest::gtest_main
  Threads::Threads)

## DISCOVERY_MODE PRE_TEST defers test enumeration to ctest run time instead of
## executing the freshly built binary during the build, which would break
## cross-compilation (the binary targets a different architecture).
gtest_discover_tests(devhostcall_test
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS "unittests")
