# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-python-rocdecode LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocdecode REQUIRED)
find_package(hip REQUIRED)

# This module drives only the rocdecode bitstream-reader API (CPU-side, in the traced
# dispatch table) to exercise ROCDECODE_API tracing without needing a working VA-API video
# decoder. It therefore links only rocdecode -- no HIP, and no RocVideoDecoder helper.
add_library(python-rocdecode SHARED)
target_sources(python-rocdecode PRIVATE python-rocdecode.cpp)
target_link_libraries(python-rocdecode PRIVATE rocprofiler-sdk::tests-build-flags
                                               rocdecode::rocdecode hip::host)
set_target_properties(
    python-rocdecode
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk"
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")
