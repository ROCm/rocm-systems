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

set(rocpdsna_sources
    ${CMAKE_CURRENT_LIST_DIR}/writer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/writer_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/insert_validator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/json_serializers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/storage_impl.cpp
)
target_sources(rocpdsna-objects PRIVATE ${rocpdsna_sources})

# ------------------------------------------------------------ #
# Common headers
# ------------------------------------------------------------ #
target_include_directories(
    rocpdsna-objects
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/common
)

add_subdirectory(queries)
add_subdirectory(data_storage)

# Link queries library to rocpdsna-objects (for compile-time includes)
target_link_libraries(rocpdsna-objects PRIVATE rocpdsna_queries)
