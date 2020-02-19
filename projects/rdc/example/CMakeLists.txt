# Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.5.0)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                             Cmake Example                                 ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})
message("--------RSMI Inc Dir: " ${RSMI_INC_DIR})
message("")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                   CACHE STRING "Location of RDC example library source code.")
# set(CMAKE_INSTALL_PREFIX "/"
#                                CACHE STRING "Default installation directory.")
# set(CPACK_PACKAGING_INSTALL_PREFIX "/"
#                                      CACHE STRING "Default packaging prefix.")
#

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -std=c++11 ")
# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif ()

set(SRC_DIR "${PROJECT_SOURCE_DIR}/example")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include")
set(LIB_BOOSTRAP_DIR "${PROJECT_BINARY_DIR}/rdc_libs")

include_directories(${INC_DIR})

set(JOBSTATS_EXAMPLE_SRC_LIST "${SRC_DIR}/job_stats_example.cc")
message("JOBSTATS_EXAMPLE_SRC_LIST=${JOBSTATS_EXAMPLE_SRC_LIST}")
set(JOBSTATS_EXAMPLE_EXE "jobstats")

link_directories(${LIB_BOOSTRAP_DIR})

add_executable(${JOBSTATS_EXAMPLE_EXE} "${JOBSTATS_EXAMPLE_SRC_LIST}")

target_link_libraries(${JOBSTATS_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(FIELDVALUE_EXAMPLE_SRC_LIST "${SRC_DIR}/field_value_example.cc")
message("FIELDVALUE_EXAMPLE_SRC_LIST=${FIELDVALUE_EXAMPLE_SRC_LIST}")
set(FIELDVALUE_EXAMPLE_EXE "fieldvalue")

link_directories(${LIB_BOOSTRAP_DIR})

add_executable(${FIELDVALUE_EXAMPLE_EXE} "${FIELDVALUE_EXAMPLE_SRC_LIST}")

target_link_libraries(${FIELDVALUE_EXAMPLE_EXE} pthread dl rdc_bootstrap)


message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Example                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
