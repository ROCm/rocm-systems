# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Configure-only test for cmake/HipFileDeprecatedAISOptions.cmake
#
# The hipfile_ais_compat test in test/CMakeLists.txt configures this project
# with every deprecated option set as -DAIS_<name>=ais-compat-AIS_<name>. The
# values are unique so a mix-up between options would be detected. This
# project doesn't declare any options, so HIPFILE_<name> will only be set if
# the compatibility layer mapped it.
#
# Remove this along with cmake/HipFileDeprecatedAISOptions.cmake.

cmake_minimum_required(VERSION 3.21)

project(hipfile_ais_compat_test LANGUAGES NONE)

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipFileDeprecatedAISOptions.cmake")

foreach(old_name IN LISTS HIPFILE_DEPRECATED_AIS_OPTIONS)
    string(REGEX REPLACE "^AIS_" "HIPFILE_" new_name "${old_name}")
    set(expected "ais-compat-${old_name}")
    if(NOT "${${new_name}}" STREQUAL "${expected}")
        message(FATAL_ERROR "${new_name} is '${${new_name}}', expected '${expected}'")
    endif()
    if(DEFINED CACHE{${old_name}})
        message(FATAL_ERROR "${old_name} was not removed from the cache")
    endif()
endforeach()
