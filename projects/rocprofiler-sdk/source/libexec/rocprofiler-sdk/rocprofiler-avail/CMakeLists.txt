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

rocprofiler_activate_clang_tidy()

add_library(rocprofv3-list-avail SHARED)
add_library(rocprofiler-sdk::rocprofv3-list-avail ALIAS rocprofv3-list-avail)
target_sources(rocprofv3-list-avail PRIVATE rocprofv3_avail.cpp)

target_link_libraries(
    rocprofv3-list-avail
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            rocprofiler-sdk::rocprofiler-sdk-perfetto
            rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-drm
            rocprofiler-sdk::rocprofiler-sdk-dl
            rocprofiler-sdk::rocprofiler-sdk-dw
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr)

set_target_properties(
    rocprofv3-list-avail
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-sdk
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

install(
    TARGETS rocprofv3-list-avail
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-sdk
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)
