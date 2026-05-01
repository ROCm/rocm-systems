# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# You may need to add -DCMAKE_PREFIX_PATH=/path/to/hipfile
# if hipFile has been installed to a non-standard location

cmake_minimum_required(VERSION 3.21)

project(aiscp LANGUAGES CXX)

find_package(hip REQUIRED CONFIG)
find_package(hipfile REQUIRED CONFIG)

add_executable(aiscp aiscp.cpp)
target_compile_features(aiscp PRIVATE cxx_std_17)
set_target_properties(aiscp PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(aiscp PRIVATE -Wall -Wextra)
target_link_libraries(aiscp PRIVATE hip::host hip::hipfile)
