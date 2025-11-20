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
# cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
# project(rocstorage)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(source)

get_filename_component(ROCSTORAGE_BASE_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
set(ROCSTORAGE_INCLUDE_DIR ${PROJECT_BINARY_DIR}/include)

foreach(_HEADER ${rocstorage_public_headers})
    string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/source/" "" _RELATIVE_PATH ${_HEADER})
    set(_DEST_FILE "${ROCSTORAGE_INCLUDE_DIR}/rocstorage/${_RELATIVE_PATH}")
    configure_file(${_HEADER} ${_DEST_FILE} COPYONLY)
endforeach()

target_include_directories(
    rocstorage
    PUBLIC
        $<BUILD_INTERFACE:${ROCSTORAGE_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

include(rocm)
include(sqlite3)
include(rocprofiler-sdk-rocpd)

target_link_libraries(rocstorage PRIVATE $<BUILD_INTERFACE:rocstorage-sqlite3>)

if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    target_link_libraries(rocstorage PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd)

    target_compile_definitions(
        rocstorage
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
    )
else()
    target_include_directories(rocstorage PRIVATE ${CMAKE_BINARY_DIR})

    target_compile_definitions(
        rocstorage
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
    )
endif()
