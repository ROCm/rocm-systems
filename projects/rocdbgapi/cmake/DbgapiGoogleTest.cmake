## SPDX-License-Identifier: MIT
## SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

# Resolve GoogleTest + GMock for the test suite.
#
# Resolution order:
#   1. If the four GTest::* targets are already defined by a parent
#      project (e.g. TheRock superbuild vendors googletest under
#      third-party/googletest), reuse them as-is.
#   2. Otherwise find_package(GTest) on the system and verify all
#      required targets exist.
#
# The all-or-nothing check on the four targets is deliberate.  A
# partial set (e.g. gtest without gmock) is rejected with a clear
# error message.  Parent projects that provide GTest must provide all
# four; TheRock does.
#
# After this module runs, the following targets are guaranteed to exist:
#   GTest::gtest
#   GTest::gtest_main
#   GTest::gmock
#   GTest::gmock_main

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
  # Parent project does not provide GoogleTest, try system installation
  message(STATUS "[dbgapi tests] Searching for system GoogleTest installation")

  find_package(GTest QUIET CONFIG)

  # Check if all required targets are available
  set(_dbgapi_gtest_system_complete TRUE)
  set(_dbgapi_gtest_missing_targets "")
  foreach(_tgt IN LISTS _dbgapi_gtest_required_targets)
    if(NOT TARGET ${_tgt})
      set(_dbgapi_gtest_system_complete FALSE)
      list(APPEND _dbgapi_gtest_missing_targets ${_tgt})
    endif()
  endforeach()

  if(_dbgapi_gtest_system_complete)
    message(STATUS "[dbgapi tests] Using system GoogleTest installation")
  else()
    # Build detailed error message
    string(REPLACE ";" ", " _missing_list "${_dbgapi_gtest_missing_targets}")
    message(FATAL_ERROR
      "[dbgapi tests] GoogleTest not found or incomplete.\n"
      "Required targets: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main\n"
      "Missing targets: ${_missing_list}\n"
      "Please install GoogleTest with GMock support:\n"
      "  Ubuntu/Debian: sudo apt-get install libgtest-dev libgmock-dev\n"
      "  Fedora/RHEL:   sudo dnf install gtest-devel gmock-devel\n"
      "Or ensure your parent project provides all four GTest::* targets.")
  endif()
endif()

