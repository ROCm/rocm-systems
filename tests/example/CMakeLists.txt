# Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake Example Lib                           ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# Set compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64 -msse -msse2")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -DDEBUG"
    CACHE STRING "Flags for Debug builds")
# note: no '-s' here unlike other CMakeLists.txt
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG"
    CACHE STRING "Flags for Release builds")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG"
    CACHE STRING "Flags for RelWithDebInfo builds")
set(CMAKE_CXX_FLAGS_MINSIZEREL
    "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG"
    CACHE STRING "Flags for MinSizeRel builds")

# Required Defines first:

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------SMI Inc Dir: " ${AMD_SMI_INCLUDE_DIR})
message("")

set(SRC_DIR "${PROJECT_SOURCE_DIR}/tests/example")
set(INC_DIR "${PROJECT_SOURCE_DIR}/client/include")

## Include common cmake modules
include(utils)

## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

set(EXAMPLE_SRC_LIST "${SRC_DIR}/rdc_client_test.cc")

message("EXAMPLE_SRC_LIST=${EXAMPLE_SRC_LIST}")

set(CLIENT_LIB_INC_LIST "${INC_DIR}/rdc_client.h")

set(TEST_CLIENT_EXE "rdc_test_client")

add_executable(${TEST_CLIENT_EXE} "${EXAMPLE_SRC_LIST}")

target_include_directories(
    ${TEST_CLIENT_EXE} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../client/include"
                               "${AMD_SMI_INCLUDE_DIR}")

target_link_libraries(${TEST_CLIENT_EXE} rdc_client)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Finished Cmake Example                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
