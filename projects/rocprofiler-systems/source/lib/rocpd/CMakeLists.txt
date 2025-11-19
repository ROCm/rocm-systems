# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.20)
project(rocpd)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(rocpd_sources ${CMAKE_CURRENT_LIST_DIR}/data_processor.cpp)
set(rocpd_headers ${CMAKE_CURRENT_LIST_DIR}/data_processor.hpp)

add_library(rocpd-library ${rocpd_sources} ${rocpd_headers})
add_subdirectory(common)
add_subdirectory(data_storage)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

include(rocm)
include(sqlite3)
include(rocprofiler-sdk-rocpd)

target_link_libraries(rocpd-library PRIVATE $<BUILD_INTERFACE:rocpd-sqlite3>)

if(rocprofiler-sdk-rocpd_FOUND)
    target_link_libraries(
        rocpd-library
        PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd
    )
endif()
