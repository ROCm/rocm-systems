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
    storage_test.cpp
    writer_test.cpp
    reader_test.cpp
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    database_test.cpp
    json_serializer_test.cpp
)

add_executable(rocstorage_unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(
    rocstorage_unit_tests
    PRIVATE
        rocstorage
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        rocstorage-sqlite3
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

target_include_directories(
    rocstorage_unit_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/source/common
)

target_compile_features(rocstorage_unit_tests PRIVATE cxx_std_17)

if(ROCSTORAGE_ENABLE_COVERAGE)
    rocstorage_add_coverage_flags(rocstorage_unit_tests)
endif()

gtest_discover_tests(
    rocstorage_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
