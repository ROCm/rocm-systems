# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# You may need to add -DCMAKE_PREFIX_PATH=/path/to/hipfile
# if hipFile has been installed to a non-standard location. That same path is
# recorded in the example binaries (via CMake's default RPATH handling), so they
# find libhipfile.so at runtime without LD_LIBRARY_PATH even from a non-standard
# install location.

cmake_minimum_required(VERSION 3.21)

project(batch_examples LANGUAGES CXX)

find_package(hip REQUIRED CONFIG)
find_package(hipfile REQUIRED CONFIG)

add_subdirectory(../common "${CMAKE_CURRENT_BINARY_DIR}/common")

add_executable(batch-roundtrip batch-roundtrip.cpp)
target_compile_features(batch-roundtrip PRIVATE cxx_std_20)
set_target_properties(batch-roundtrip PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(batch-roundtrip PRIVATE -Wall -Wextra)
target_link_libraries(batch-roundtrip PRIVATE examples_common)
