// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/output/sql/lz4_extension.hpp"

#include <sqlite3.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace sql = rocprofiler::tool::sql;

TEST(lz4_extension, buffer_round_trip)
{
    const auto input = std::string(1024, 'x') + R"({"key":"value","nested":[1,2,3]})";
    const auto blob  = sql::lz4_compress_buffer(input);

    ASSERT_GT(blob.size(), 8);
    EXPECT_EQ(sql::lz4_decompress_buffer({reinterpret_cast<const char*>(blob.data()), blob.size()}),
              input);
}

TEST(lz4_extension, empty_round_trip)
{
    auto blob = sql::lz4_compress_buffer({});
    EXPECT_TRUE(blob.empty());
    EXPECT_TRUE(sql::lz4_decompress_buffer({}).empty());
}

TEST(lz4_extension, invalid_magic_throws)
{
    EXPECT_THROW(sql::lz4_decompress_buffer("not-lz4"), std::runtime_error);
}

TEST(lz4_extension, sqlite_udf_round_trip)
{
    sqlite3* conn = nullptr;
    ASSERT_EQ(sqlite3_open(":memory:", &conn), SQLITE_OK);
    ASSERT_NE(conn, nullptr);

    sql::register_lz4_functions(conn);

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(sqlite3_prepare_v2(conn,
                                 "SELECT CAST(lz4_decompress(lz4_compress('{\"a\":1}')) AS TEXT)",
                                 -1,
                                 &stmt,
                                 nullptr),
              SQLITE_OK);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    EXPECT_STREQ(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)), "{\"a\":1}");
    EXPECT_EQ(sqlite3_finalize(stmt), SQLITE_OK);
    EXPECT_EQ(sqlite3_close(conn), SQLITE_OK);
}
