# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# rocprofv3-collector: rocprofv3's out-of-process path (rocprofv3 --collector). It runs
# the application under the collector, then writes rocprofv3's output files from the
# records.
rocprofiler_activate_clang_tidy()

add_executable(rocprofv3-collector)
target_sources(rocprofv3-collector PRIVATE main.cpp)
target_include_directories(rocprofv3-collector
                           PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)
target_link_libraries(
    rocprofv3-collector
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-collector-shared-library
            rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-sqlite3
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cereal)
set_target_properties(
    rocprofv3-collector
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-sdk
               BUILD_RPATH "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}")

install(
    TARGETS rocprofv3-collector
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-sdk
    COMPONENT tools)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
