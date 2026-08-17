# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

project(ROCclr)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(AUTHOR_WARNING "ROCclr is being built as a standalone project. This isn't supported anymore.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(ROCclr)

# BUILD_TESTS is declared -- and enable_testing() runs -- at the CLR root so
# that `ctest` discovers these tests from the top-level build directory.
if(BUILD_TESTS)
  add_subdirectory(tests/unit)
endif()
