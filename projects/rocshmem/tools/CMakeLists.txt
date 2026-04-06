###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################
# SOURCES
###############################################################################

# Embed rocshmem_config.h as a C++17 raw string literal at configure time so
# rocshmem_info does not need to locate the file at runtime.
set(_config_h "${CMAKE_BINARY_DIR}/include/rocshmem/rocshmem_config.h")
file(READ "${_config_h}" _config_content)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rocshmem_config_embedded.hpp"
  "// Auto-generated from rocshmem_config.h — do not edit.\n"
  "#pragma once\n"
  "static const char rocshmem_config_h_content[] = R\"ROCSHMEM_CFG_END(\n"
  "${_config_content}"
  ")ROCSHMEM_CFG_END\";\n"
)

add_executable(rocshmem_info rocshmem_info.cpp)

target_include_directories(
  rocshmem_info
  PRIVATE
    roc::rocshmem
    "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(
  rocshmem_info
  PRIVATE
    roc::rocshmem
)
