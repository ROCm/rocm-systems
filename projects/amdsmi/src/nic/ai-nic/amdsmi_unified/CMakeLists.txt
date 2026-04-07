#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.16)

project(amdsminic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(NIC_INCLUDE_DIR ${PROJECT_ROOT}/inc ${PROJECT_ROOT}/interface)
set(NIC_SOURCE_DIR ${PROJECT_ROOT}/src)
set(NIC_BUILD_DIR ${CMAKE_BINARY_DIR}/build)
set(NIC_INTERFACE_DIR ${CMAKE_BINARY_DIR}/interface)

set(NIC_DEFAULT_CXX_FLAGS
    "-Wall -Wextra -Werror -Wno-missing-field-initializers -Wno-array-bounds -Wmissing-declarations -Werror=conversion -Wshift-negative-value -fPIC"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NIC_DEFAULT_CXX_FLAGS}")

# Linker hardening flags - applied only at link time, not during compilation.
# These were previously in CMAKE_CXX_FLAGS which caused -Wunused-command-line-argument
# errors with clang-based compilers (e.g. amdclang).
set(NIC_LINKER_FLAGS "-Wl,-z,relro,-z,noexecstack,-z,noexecheap -Wl,--strip-debug -Wl,--strip-all")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NIC_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${NIC_LINKER_FLAGS}")

file(GLOB CPP_SRCS "${NIC_SOURCE_DIR}/*.cpp")
add_library(amdsminic STATIC ${CPP_SRCS})
set_target_properties(amdsminic PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${NIC_BUILD_DIR})
target_include_directories(
    amdsminic
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_ROOT}/inc>"
        "$<BUILD_INTERFACE:${PROJECT_ROOT}/interface>"
        "$<BUILD_INTERFACE:${NIC_INTERFACE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/inc>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/interface>"
)
