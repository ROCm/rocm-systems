## SPDX-License-Identifier: MIT
## SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

include_guard(GLOBAL)

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

# Helper functions for declaring rocdbgapi tests.
#
# These wrap add_executable / add_test so each test gets a consistent
# set of properties (C++ standard, GoogleTest link, CTest labels,
# working directory) without each CMakeLists.txt repeating the same
# boilerplate.
#
# Public API:
#   add_dbgapi_unit_test    (name SOURCES ... [LABELS ...] [LIBS ...])
#
# All tests get the "dbgapi" and "unit" labels automatically.
# Internal functions exist to make it easier to add other test
# categories later.

# Single source of truth for OS labels so they don't drift between
# helpers.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_DBGAPI_OS_LABEL "os-linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(_DBGAPI_OS_LABEL "os-windows")
else()
  set(_DBGAPI_OS_LABEL "")
endif()

# Internal: configure common properties shared by all test executables.
function(_dbgapi_apply_common_test_properties target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON)

  if(MSVC)
    # rocdbgapi disables RTTI in the main library but test code needs
    # RTTI for GMock and gtest's death-test machinery.  Don't propagate
    # the library's /GR- here; the default /GR is what we want.
    target_compile_options(${target} PRIVATE /permissive-)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wshadow -Wno-attributes)
  endif()
endfunction()

# add_dbgapi_unit_test(name
#                     SOURCES src1.cpp [src2.cpp ...]
#                     [LABELS extra-label ...]
#                     [LIBS  extra::link ...])
function(add_dbgapi_unit_test target_name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LABELS;LIBS" ${ARGN})

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_dbgapi_unit_test(${target_name}): SOURCES is required")
  endif()

  add_executable(${target_name} ${ARG_SOURCES})
  _dbgapi_apply_common_test_properties(${target_name})

  target_link_libraries(${target_name} PRIVATE
    GTest::gtest_main
    GTest::gmock
    ${ARG_LIBS})

  set(labels "dbgapi" "unit")
  if(_DBGAPI_OS_LABEL)
    list(APPEND labels ${_DBGAPI_OS_LABEL})
  endif()
  if(ARG_LABELS)
    list(APPEND labels ${ARG_LABELS})
  endif()
  list(REMOVE_DUPLICATES labels)

  add_test(NAME ${target_name} COMMAND ${target_name})
  set_tests_properties(${target_name} PROPERTIES
    LABELS "${labels}"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${target_name}>)
endfunction()

add_subdirectory(unit)

