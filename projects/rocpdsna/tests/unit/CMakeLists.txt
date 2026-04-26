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
    writer_test.cpp
    reader_test.cpp
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    table_select_query_test.cpp
    database_test.cpp
    json_serializer_test.cpp
    concurrent_flush_test.cpp
)

add_executable(rocpdsna_unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(
    rocpdsna_unit_tests
    PRIVATE
        rocpdsna
        rocpdsna_queries
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        rocpdsna-sqlite3
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/rocpd.db
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db
    COPYONLY
)

set(ROCPD_DB_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db)

target_compile_definitions(
    rocpdsna_unit_tests
    PUBLIC ROCPD_DB_PATH="${ROCPD_DB_PATH}"
)

target_include_directories(
    rocpdsna_unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/common
        ${CMAKE_SOURCE_DIR}/source/data_storage
        ${CMAKE_SOURCE_DIR}/source/queries
)

target_compile_features(rocpdsna_unit_tests PRIVATE cxx_std_17)

if(ROCPDSNA_ENABLE_COVERAGE)
    rocpdsna_add_coverage_flags(rocpdsna_unit_tests)
endif()

gtest_discover_tests(
    rocpdsna_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
