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
project(amdcuid)

# Set platform-appropriate default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/Program Files/AMD/AMDCUID" CACHE PATH "Installation prefix" FORCE)
    else()  # Linux/Unix
        set(CMAKE_INSTALL_PREFIX "/opt/amdcuid" CACHE PATH "Installation prefix" FORCE)
    endif()
endif()

include(GNUInstallDirs)

if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)  # DLLs
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  # Import libs
endif()

# Platform-specific dependencies; If we decide to use the default Windows crypto API, we can remove the OpenSSL dependency on Windows
if(WIN32)
    # Windows may need different OpenSSL path
    set(OPENSSL_ROOT_DIR "C:/OpenSSL-Win64" CACHE PATH "OpenSSL location")
endif()

add_subdirectory(lib)
add_subdirectory(cli)
add_subdirectory(daemon)
add_subdirectory(example)
add_subdirectory(tests)
