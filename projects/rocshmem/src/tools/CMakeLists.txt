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

execute_process (
  COMMAND bash -c "git -C ${PROJECT_SOURCE_DIR} rev-parse HEAD;"
  OUTPUT_VARIABLE GIT_HASH
)

string(STRIP "${GIT_HASH}" GIT_HASH)
string(REPLACE ";" " " ROCSHMEM_OFFLOAD_TARGETS "${COMPILING_TARGETS}")

add_executable(rocshmem_info rocshmem_info.cpp)

target_compile_definitions(
  rocshmem_info
  PRIVATE
  ROCSHMEM_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
  ROCSHMEM_GIT_HASH="${GIT_HASH}"
  ROCSHMEM_OFFLOAD_TARGETS="${ROCSHMEM_OFFLOAD_TARGETS}"
)

target_include_directories(
  rocshmem_info
  PRIVATE
    roc::rocshmem
)

target_link_libraries(
  rocshmem_info
  PRIVATE
    roc::rocshmem
)
