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

# CPU-only unit tests: they run without a GPU, but compile the production
# SelectSurfaceFormat() from roc_video_dec.h, which pulls in the rocdecode API
# and HIP headers -- so the ROCm toolchain is required to build them. Wired into
# the rocdecode test build (see ../CMakeLists.txt) and buildable standalone
# (with ROCM_PATH on CMAKE_PREFIX_PATH):
#   cmake -S . -B build && cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.18)

# Only declare our own project() when configured standalone; when pulled in via
# add_subdirectory() from the rocdecode test build we inherit the parent's, and
# the C++ standard is set on the target (below) rather than via directory-wide
# variables to avoid affecting the parent configuration.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocdecode_unit_tests LANGUAGES CXX)
endif()

find_package(GTest REQUIRED)
find_package(HIP REQUIRED)
include(GoogleTest)
enable_testing()

add_executable(surface_format_selection_test surface_format_selection_test.cpp)
target_compile_features(surface_format_selection_test PRIVATE cxx_std_17)
target_include_directories(surface_format_selection_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode
    ${CMAKE_CURRENT_SOURCE_DIR}/../../api)
target_link_libraries(surface_format_selection_test PRIVATE GTest::gtest_main hip::host)
gtest_discover_tests(surface_format_selection_test)
