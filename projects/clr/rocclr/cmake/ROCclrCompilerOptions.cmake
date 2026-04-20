# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

include_guard()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if (CMAKE_VERSION VERSION_LESS "3.20")
    # This code is neccessary to avoid this command line warning:
    # "Overriding /GR with /GR- cl: command line warning D9025"
    #
    # /GR is implied by MSVC anyway. So getting rid of it doesn't matter.
    string(REPLACE "/GR" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif()
endif()
