# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# You may need to add -DCMAKE_PREFIX_PATH=/path/to/hipfile
# if hipFile has been installed to a non-standard location. That same path is
# recorded in the example binaries (via CMake's default RPATH handling), so they
# find libhipfile.so at runtime without LD_LIBRARY_PATH even from a non-standard
# install location.

cmake_minimum_required(VERSION 3.21)

project(async_examples LANGUAGES CXX)

find_package(hip REQUIRED CONFIG)
find_package(hipfile REQUIRED CONFIG)

# The async examples share helpers from the common/ directory. Build that
# static library here so the examples can link it without duplicating the code.
add_subdirectory(../common "${CMAKE_CURRENT_BINARY_DIR}/common")

set(ASYNC_EXAMPLES
    roundtrip-async
    roundtrip-async-nonblocking-stream
    roundtrip-async-multi-stream
    roundtrip-async-multi-stream-registered
)

foreach(example ${ASYNC_EXAMPLES})
    add_executable(${example} "${example}.cpp")
    target_compile_features(${example} PRIVATE cxx_std_17)
    set_target_properties(${example} PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_options(${example} PRIVATE -Wall -Wextra)
    target_link_libraries(${example} PRIVATE examples_common)
endforeach()
