# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

# ------------------------------------------------------------ #
# rocpdsna_queries static library
# ------------------------------------------------------------ #

add_library(rocpdsna_queries STATIC)

set(queries_sources
    ${CMAKE_CURRENT_LIST_DIR}/query_builder_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/select/select_query_builders.cpp
)

target_sources(rocpdsna_queries PRIVATE ${queries_sources})

target_include_directories(
    rocpdsna_queries
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
)

# Include common headers for traits
target_include_directories(
    rocpdsna_queries
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_compile_options(
    rocpdsna_queries
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall
        -Wextra
        -Wshadow
        -Wvla
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wnull-dereference
        >
)

set_target_properties(
    rocpdsna_queries
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
)
