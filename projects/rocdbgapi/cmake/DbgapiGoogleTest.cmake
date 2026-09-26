## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# Resolve GoogleTest + GMock for the test suite.
#
# Resolution order:
#   1. If the four GTest::* targets are already defined by a parent
#      project (e.g. TheRock superbuild vendors googletest under
#      third-party/googletest), reuse them as-is.
#   2. Otherwise FetchContent a pinned release.
#
# The all-or-nothing check on the four targets is deliberate.  A
# partial set (e.g. gtest without gmock) would otherwise collide with
# the FetchContent fallback when it redeclared the targets that
# already exist.  Parent projects that provide GTest must provide all
# four; TheRock does.
#
# We do not fall back to find_package(GTest) on the system.  Many
# distros ship a GTest CMake config that exports gtest only (no
# gmock), Ubuntu 22.04 ships an older release than we pin to, and a
# partial system package mixing with FetchContent produces
# multiple-definition link errors.  A future commit may add an
# opt-in DBGAPI_USE_SYSTEM_GTEST=ON for packagers who need it.
#
# After this module runs, the following targets are guaranteed to exist:
#   GTest::gtest
#   GTest::gtest_main
#   GTest::gmock
#   GTest::gmock_main
#
# MSVC note: GoogleTest defaults to building against the static CRT
# (/MT), while rocdbgapi and most consumers build against the shared
# CRT (/MD).  We force GoogleTest to match by setting
# gtest_force_shared_crt=ON before the subdirectory is added.  See
# https://google.github.io/googletest/quickstart-cmake.html

include_guard(GLOBAL)

set(_dbgapi_gtest_required_targets
  GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

set(_dbgapi_gtest_have_all TRUE)
foreach(_tgt IN LISTS _dbgapi_gtest_required_targets)
  if(NOT TARGET ${_tgt})
    set(_dbgapi_gtest_have_all FALSE)
    break()
  endif()
endforeach()

if(_dbgapi_gtest_have_all)
  message(STATUS
    "[dbgapi tests] Using GoogleTest targets provided by parent project")
else()
  # Pinned to a release tag for reproducibility.  Bump deliberately; do
  # not track main.
  set(DBGAPI_GTEST_TAG "v1.15.2"
    CACHE STRING "GoogleTest release tag to fetch")
  mark_as_advanced(DBGAPI_GTEST_TAG)

  message(STATUS "[dbgapi tests] Fetching GoogleTest ${DBGAPI_GTEST_TAG}")

  include(FetchContent)

  # Required for MSVC builds where rocdbgapi uses the shared CRT (/MD).
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  # We don't install GoogleTest with the project.
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

  # Force GoogleTest to build as static archives even when the parent
  # project has BUILD_SHARED_LIBS=ON (rocdbgapi's default).  A shared
  # libgtest is fragile here: the test binary's per-test static
  # constructors can run before libgtest.so's own singleton init,
  # which leaves gtest's internal unordered_map with a zero bucket
  # count and crashes with SIGFPE inside std::hash bucket math.
  # Linking GTest in statically eliminates the cross-image init order
  # problem entirely.
  set(_dbgapi_saved_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        ${DBGAPI_GTEST_TAG}
    GIT_SHALLOW    TRUE)

  FetchContent_MakeAvailable(googletest)

  set(BUILD_SHARED_LIBS ${_dbgapi_saved_BUILD_SHARED_LIBS})
endif()

# Sanity check the targets are present so a misconfigured package
# fails loudly here rather than at link time in every test.
foreach(_tgt IN LISTS _dbgapi_gtest_required_targets)
  if(NOT TARGET ${_tgt})
    message(FATAL_ERROR
      "[dbgapi tests] Expected target ${_tgt} after GoogleTest resolution")
  endif()
endforeach()
