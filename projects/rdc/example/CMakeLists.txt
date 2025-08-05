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

# WARN: This is a standalone CMake project! Do not include as a subdirectory!

# This project is meant to demo RDC library
# See README.md for more information

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.15)

option(CMAKE_VERBOSE_MAKEFILE "Enable verbose output" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands for linters and autocompleters" ON)

# Set compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64 -msse -msse2")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -DDEBUG"
    CACHE STRING
    "Flags for Debug builds"
)
# note: no '-s' here unlike other CMakeLists.txt
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG"
    CACHE STRING
    "Flags for Release builds"
)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG"
    CACHE STRING
    "Flags for RelWithDebInfo builds"
)
set(CMAKE_CXX_FLAGS_MINSIZEREL
    "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG"
    CACHE STRING
    "Flags for MinSizeRel builds"
)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4.0)
    message("Compiler version is " ${CMAKE_CXX_COMPILER_VERSION})
    message(FATAL_ERROR "Require at least gcc-5.4.0")
endif()

project(RDC_example)

# provides cmake_print_variables(VAR)
include(CMakePrintHelpers)

# required variables
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_DIR "$ENV{ROCM_PATH}" CACHE STRING "ROCm directory.")
else()
    set(ROCM_DIR "/opt/rocm" CACHE STRING "ROCm directory.")
endif()

# add package search paths
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ROCM_DIR})
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${ROCM_DIR}/lib ${ROCM_DIR}/lib64)

# RDC provides librdc_bootstrap
find_package(rdc CONFIG REQUIRED)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                             Cmake Example                                 ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------------ROCM_DIR: " ${ROCM_DIR})
message("")

set(JOBSTATS_EXAMPLE_SRC_LIST "job_stats_example.cc")
cmake_print_variables(JOBSTATS_EXAMPLE_SRC_LIST)
set(JOBSTATS_EXAMPLE_EXE "jobstats")
add_executable(${JOBSTATS_EXAMPLE_EXE} "${JOBSTATS_EXAMPLE_SRC_LIST}")
target_link_libraries(${JOBSTATS_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(FIELDVALUE_EXAMPLE_SRC_LIST "field_value_example.cc")
cmake_print_variables(FIELDVALUE_EXAMPLE_SRC_LIST)
set(FIELDVALUE_EXAMPLE_EXE "fieldvalue")
add_executable(${FIELDVALUE_EXAMPLE_EXE} "${FIELDVALUE_EXAMPLE_SRC_LIST}")
target_link_libraries(${FIELDVALUE_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(DIAGNOSTIC_EXAMPLE_SRC_LIST "diagnostic_example.cc")
cmake_print_variables(DIAGNOSTIC_EXAMPLE_SRC_LIST)
set(DIAGNOSTIC_EXAMPLE_EXE "diagnostic")
add_executable(${DIAGNOSTIC_EXAMPLE_EXE} "${DIAGNOSTIC_EXAMPLE_SRC_LIST}")
target_link_libraries(${DIAGNOSTIC_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(ROCPROFILER_EXAMPLE_SRC_LIST "rocprofiler_example.cc")
cmake_print_variables(ROCPROFILER_EXAMPLE_SRC_LIST)
set(ROCPROFILER_EXAMPLE_EXE "rocprofiler")
add_executable(${ROCPROFILER_EXAMPLE_EXE} "${ROCPROFILER_EXAMPLE_SRC_LIST}")
target_link_libraries(${ROCPROFILER_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(POLICY_EXAMPLE_SRC_LIST "policy_example.cc")
cmake_print_variables(POLICY_EXAMPLE_SRC_LIST)
set(POLICY_EXAMPLE_EXE "policy")
add_executable(${POLICY_EXAMPLE_EXE} "${POLICY_EXAMPLE_SRC_LIST}")
target_link_libraries(${POLICY_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(HEALTH_EXAMPLE_SRC_LIST "health_example.cc")
cmake_print_variables(HEALTH_EXAMPLE_SRC_LIST)
set(HEALTH_EXAMPLE_EXE "health")
add_executable(${HEALTH_EXAMPLE_EXE} "${HEALTH_EXAMPLE_SRC_LIST}")
target_link_libraries(${HEALTH_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(CONFIG_EXAMPLE_SRC_LIST "config_example.cc")
cmake_print_variables(CONFIG_EXAMPLE_SRC_LIST)
set(CONFIG_EXAMPLE_EXE "config")
add_executable(${CONFIG_EXAMPLE_EXE} "${CONFIG_EXAMPLE_SRC_LIST}")
target_link_libraries(${CONFIG_EXAMPLE_EXE} pthread dl rdc_bootstrap)

set(TOPOLOGYLINK_EXAMPLE_SRC_LIST "topologylink_example.cc")
cmake_print_variables(TOPOLOGYLINK_EXAMPLE_SRC_LIST)
set(TOPOLOGYLINK_EXAMPLE_EXE "topologylink")
add_executable(${TOPOLOGYLINK_EXAMPLE_EXE} "${TOPOLOGYLINK_EXAMPLE_SRC_LIST}")
target_link_libraries(${TOPOLOGYLINK_EXAMPLE_EXE} pthread dl rdc_bootstrap)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Example                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
