################################################################################
##
## Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
##
## SPDX-License-Identifier: MIT
##
################################################################################

## Unit tests for the header-only utilities in core/util.
## Only lazy_ptr.h is exercised today; add sources here as more are covered.

## gtest_discover_tests(... DISCOVERY_MODE PRE_TEST) requires CMake >= 3.18.
## The user opted in via BUILD_ROCR_TESTS, so fail loudly rather than silently
## producing no tests.
if (CMAKE_VERSION VERSION_LESS 3.18)
  message(FATAL_ERROR
    "BUILD_ROCR_TESTS requires CMake >= 3.18 "
    "(gtest_discover_tests DISCOVERY_MODE PRE_TEST); found ${CMAKE_VERSION}.")
endif()

## Threads::Threads is used explicitly (lazy_ptr uses std::mutex) and is also a
## transitive dependency of GTest::gtest_main.
find_package(Threads REQUIRED)

## Locate GTest, or fetch it if the build image doesn't provide one. Because the
## test is opt-in (BUILD_ROCR_TESTS), a missing GTest must NOT silently skip the
## regression test — fall back to FetchContent, mirroring projects/aqlprofile.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "hsa-runtime64: GTest not found on system -- fetching via FetchContent")
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

add_executable(lazy_ptr_test lazy_ptr_test.cpp)

## The test only needs the header (core/util/lazy_ptr.h and its relative
## includes), which resolve from the hsa-runtime source root. No link against
## the runtime library is required.
target_include_directories(lazy_ptr_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(lazy_ptr_test PRIVATE
  GTest::gtest
  GTest::gtest_main
  Threads::Threads)

## DISCOVERY_MODE PRE_TEST defers test enumeration to ctest run time instead of
## executing the freshly built binary during the build, which would break
## cross-compilation (the binary targets a different architecture).
gtest_discover_tests(lazy_ptr_test
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS "unittests")
