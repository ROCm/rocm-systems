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

include(gtest)

set(UNIT_TEST_SOURCES
    trace_test.cpp
    database_test.cpp
    c_interface_test.cpp
)

add_executable(rocstorage_reader_unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(
    rocstorage_reader_unit_tests
    PRIVATE
        rocstorage-reader
        rocstorage
        GTest::gtest
        GTest::gtest_main
        rocstorage-sqlite3
        spdlog::spdlog
)

target_include_directories(
    rocstorage_reader_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../source/datamodel
        ${CMAKE_CURRENT_SOURCE_DIR}/../../source/database
)

target_compile_features(rocstorage_reader_unit_tests PRIVATE cxx_std_17)

gtest_discover_tests(
    rocstorage_reader_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
