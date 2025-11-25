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

# TODO: Remove block below, this is for local development only
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(rocstorage)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------ #
# Library
# ------------------------------------------------------------ #
add_library(rocstorage SHARED)
add_subdirectory(source)

# ------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------ #
set(rocstorage_public_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/writer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/storage.hpp
)

foreach(HEADER_FILE ${rocstorage_public_headers})
    file(RELATIVE_PATH HEADER_REL ${CMAKE_CURRENT_SOURCE_DIR}/include ${HEADER_FILE})
    configure_file(
        ${HEADER_FILE}
        ${CMAKE_BINARY_DIR}/include/rocstorage/${HEADER_REL}
        COPYONLY
    )
endforeach()

target_include_directories(
    rocstorage
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ------------------------------------------------------------ #
# CMake modules
# ------------------------------------------------------------ #
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

# ------------------------------------------------------------ #
# SQLite3
# ------------------------------------------------------------ #
include(sqlite3)
target_link_libraries(rocstorage PRIVATE $<BUILD_INTERFACE:rocstorage-sqlite3>)

# ------------------------------------------------------------ #
# Schema from rocprofiler-sdk-rocpd
# ------------------------------------------------------------ #
set(SQL_SCHEMA_DIR ${CMAKE_CURRENT_LIST_DIR}/source/data_storage/schema)
set(SQL_SCHEMA_BINARY_DIR ${CMAKE_BINARY_DIR}/source/data_storage/)

include(rocm)
include(rocprofiler-sdk-rocpd)

if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    target_link_libraries(rocstorage PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd)

    target_compile_definitions(
        rocstorage
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
    )
else()
    target_include_directories(rocstorage PRIVATE ${SQL_SCHEMA_BINARY_DIR})

    target_compile_definitions(
        rocstorage
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
    )
endif()

# ------------------------------------------------------------ #
# CMake packaging
# ------------------------------------------------------------ #
