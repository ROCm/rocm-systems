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
project(amdcuid_tool)

set(CMAKE_CXX_STANDARD 14)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Add the tool executable
add_executable(amdcuid_tool amdcuid_tool.cc)

# Link against the CUID library (public API only)
# The tool now uses only the public API defined in amd_cuid.h
target_link_libraries(amdcuid_tool amdcuid_static OpenSSL::Crypto)

# Include directories - only need access to public headers
target_include_directories(amdcuid_tool PRIVATE ${CMAKE_SOURCE_DIR}/lib)

# Install the tool
install(TARGETS amdcuid_tool DESTINATION ${CMAKE_INSTALL_BINDIR})
