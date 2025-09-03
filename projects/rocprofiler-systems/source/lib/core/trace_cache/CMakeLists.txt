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

set(trace_cache_sources
    ${CMAKE_CURRENT_LIST_DIR}/cache_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/storage_parser.cpp
    ${CMAKE_CURRENT_LIST_DIR}/buffer_storage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/metadata_registry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rocpd_post_processing.cpp
)

set(trace_cache_headers
    ${CMAKE_CURRENT_LIST_DIR}/cache_manager.hpp
    ${CMAKE_CURRENT_LIST_DIR}/storage_parser.hpp
    ${CMAKE_CURRENT_LIST_DIR}/buffer_storage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cache_utility.hpp
    ${CMAKE_CURRENT_LIST_DIR}/metadata_registry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocpd_post_processing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_type.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${trace_cache_sources} ${trace_cache_headers}
)
