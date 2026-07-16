# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# You may need to add -DCMAKE_PREFIX_PATH=/path/to/hipfile
# if hipFile has been installed to a non-standard location

cmake_minimum_required(VERSION 3.21)

project(examples_common LANGUAGES CXX)

find_package(hip REQUIRED CONFIG)
find_package(hipfile REQUIRED CONFIG)

# Shared helpers used by the basics/ and async/ examples. The dependent example
# directories pull this in via add_subdirectory(), so the include dir and the
# hipFile link are PUBLIC to propagate to them.
add_library(examples_common STATIC examples_common.cpp)
target_compile_features(examples_common PRIVATE cxx_std_17)
set_target_properties(examples_common PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(examples_common PRIVATE -Wall -Wextra)
target_include_directories(examples_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(examples_common PUBLIC hip::host hip::hipfile)
