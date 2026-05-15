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
project(amdcuid_lib)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenSSL REQUIRED)

set(CUID_SOURCES
    src/cuid.cc
    src/cuid_util.cc
    src/cuid_device.cc
    src/cuid_device_manager.cc
    src/cuid_cpu.cc
    src/cuid_gpu.cc
    src/cuid_nic.cc
    src/cuid_platform.cc
    src/cuid_file.cc
    src/hmac.cc
    src/pci_util.cc
    src/acpi_parser.cc
    src/smbios_util.cc
)

set(CUID_HEADERS
    src/cuid_util.h
    src/cuid_device.h
    src/cuid_device_manager.h
    src/cuid_cpu.h
    src/cuid_gpu.h
    src/cuid_nic.h
    src/cuid_platform.h
    src/cuid_file.h
    src/hmac.h
    src/ipc_protocol.h
    src/pci_util.h
    src/acpi_parser.h
    src/smbios_util.h
    src/cuid_internal.h
)

add_library(amdcuid_static STATIC ${CUID_SOURCES} ${CUID_HEADERS})
add_library(amdcuid_shared SHARED ${CUID_SOURCES} ${CUID_HEADERS})

target_include_directories(amdcuid_static 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(amdcuid_shared 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Shared library
install(TARGETS amdcuid_shared
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # Windows DLLs go here
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    # Linux .so
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    # Static libs & Windows import libs
    COMPONENT library
)

# Static library
install(TARGETS amdcuid_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT library
)

# Install public headers
set(PUBLIC_HEADERS
    include/amd_cuid.h
)
install(FILES ${PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT library)
