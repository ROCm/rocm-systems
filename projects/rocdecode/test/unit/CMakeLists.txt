################################################################################
# Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# Unit tests for rocdecode — pure logic tests that do NOT require GPU hardware.
# These test extracted helper functions (e.g. PopulateLayerPointers) that were
# previously embedded in hardware-dependent code paths.

cmake_minimum_required(VERSION 3.10)
project(rocdecode_unit_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

# video_layer_utils.h only needs the rocdecode public API header (for rocDecStatus).
# No HIP, VA-API, or GPU hardware required.
include_directories(
    ${PROJECT_SOURCE_DIR}/../../src/rocdecode
    ${PROJECT_SOURCE_DIR}/../../api
)

add_executable(rocdecode_unit_tests
    populate_layer_pointers_test.cpp
)

target_link_libraries(rocdecode_unit_tests
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(rocdecode_unit_tests)
