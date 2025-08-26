# Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All Rights Reserved.
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

#-------------------------------------elf_test--------------------------------------#
# This is unit test for amd::Elf.
# The test is on top of rocclr, so rocclr must be built and installed firstly.
# This file is seperate from cmake file of rocclr to prevent interference. 

find_package(amd_comgr REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/amd_comgr
    lib/cmake/amd_comgr)

find_package(hsa-runtime64 REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/hsa-runtime64)

find_package(Threads REQUIRED)

# Look for ROCclr which contains elfio
find_package(ROCclr REQUIRED CONFIG
  PATHS
    /opt/rocm
    /opt/rocm/rocclr)

add_executable(elf_test main.cpp)
set_target_properties(
    elf_test PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_include_directories(elf_test
  PRIVATE
    $<TARGET_PROPERTY:amdrocclr_static,INTERFACE_INCLUDE_DIRECTORIES>)

add_definitions(-DUSE_COMGR_LIBRARY -DCOMGR_DYN_DLL -DWITH_LIGHTNING_COMPILER -DDEBUG)

target_link_libraries(elf_test PRIVATE amdrocclr_static)

#-------------------------------------elf_test--------------------------------------#