# ##############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.15)

# Built in-tree from the top-level CMakeLists (BUILD_FROM_SOURCE), where the
# rocencode target and the rocenccaps sample already exist.
if(NOT TARGET rocencode)
  message(STATUS "test/CMakeLists.txt: rocencode target not present; no tests registered.")
  return()
endif()

add_executable(rocencode_api_negative_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/rocEncodeNegativeApiTests/rocencode_api_negative_tests.cpp)
target_include_directories(rocencode_api_negative_tests
  PRIVATE ${CMAKE_SOURCE_DIR}/api)
target_link_libraries(rocencode_api_negative_tests rocencode)

# rocEncode_Caps
#
# Runs the capability sample across every device. It exits 0 both when an
# encoder is found and when none is -- a decode-only part (Instinct MI300-class,
# Navi 24 / VCN 3.0.33) must report no-encode cleanly rather than crash, and
# that is exactly what this test pins down.
add_test(NAME rocEncode_Caps
         COMMAND $<TARGET_FILE:rocenccaps>
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(rocEncode_Caps PROPERTIES
  TIMEOUT 120
  FAIL_REGULAR_EXPRESSION "query failed")

# rocEncode_Negative_API
#
# Bad arguments, bad versions, unsupported device types and every
# declared-but-unimplemented entry point. Must pass on hardware with no encoder.
add_test(NAME rocEncode_Negative_API
         COMMAND $<TARGET_FILE:rocencode_api_negative_tests>
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(rocEncode_Negative_API PROPERTIES
  TIMEOUT 120
  FAIL_REGULAR_EXPRESSION "FAIL:")
