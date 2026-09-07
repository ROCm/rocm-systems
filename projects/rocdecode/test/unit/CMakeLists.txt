# ##############################################################################
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
# ##############################################################################

# Host-only unit tests (no HIP / no GPU required). They compile the production
# surface_format_utils.h in ROCDECODE_SURFACE_FORMAT_UTILS_STANDALONE mode, so
# the same SelectSurfaceFormat() the decoder uses is exercised directly. Wired
# into the rocdecode test build (see ../CMakeLists.txt) and buildable standalone:
#   cmake -S . -B build && cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.18)
project(rocdecode_unit_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
include(GoogleTest)
enable_testing()

add_executable(surface_format_selection_test surface_format_selection_test.cpp)
target_include_directories(surface_format_selection_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode)
target_compile_definitions(surface_format_selection_test PRIVATE
    ROCDECODE_SURFACE_FORMAT_UTILS_STANDALONE)
target_link_libraries(surface_format_selection_test PRIVATE GTest::gtest_main)
gtest_discover_tests(surface_format_selection_test)
