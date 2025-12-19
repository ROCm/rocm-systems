# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10)
project(cuid_lib)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenSSL REQUIRED)

set(CUID_SOURCES
    cuid.cc
    cuid_util.cc
    cuid_device.cc
    cuid_device_manager.cc
    cuid_cpu.cc
    cuid_gpu.cc
    cuid_nic.cc
    cuid_platform.cc
    cuid_file.cc
    hmac.cc
    cuid_lib_loader.cc
    pci_util.cc
    acpi_parser.cc
    smbios_util.cc
)

set(CUID_HEADERS
    cuid.h
    cuid_util.h
    cuid_device.h
    cuid_device_manager.h
    cuid_cpu.h
    cuid_gpu.h
    cuid_nic.h
    cuid_platform.h
    cuid_file.h
    hmac.h
    cuid_lib_loader.h
    pci_util.h
    acpi_parser.h
    smbios_util.h
)

add_library(cuid_static STATIC ${CUID_SOURCES} ${CUID_HEADERS})
add_library(cuid_shared SHARED ${CUID_SOURCES} ${CUID_HEADERS})

target_include_directories(cuid_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cuid_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
