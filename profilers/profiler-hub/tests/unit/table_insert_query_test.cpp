// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "insert/table_insert_query.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace
{

using namespace profiler_hub::queries::insert;

class table_insert_query_test : public ::testing::Test
{
protected:
    table_insert_query m_query;
};

TEST_F(table_insert_query_test, simple_insert_with_single_column)
{
    auto query_string = m_query.set_table_name("test_table")
                            .set_columns("id")
                            .set_values(1)
                            .get_query_string();
    EXPECT_EQ(query_string, "INSERT INTO test_table ( id ) VALUES ( 1 )");
}

TEST_F(table_insert_query_test, insert_with_multiple_columns_and_values)
{
    auto query_string = m_query.set_table_name("users")
                            .set_columns("id", "name", "age")
                            .set_values(1, "Alice", 30)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "INSERT INTO users ( id, name, age ) VALUES ( 1, \"Alice\", 30 )");
}

TEST_F(table_insert_query_test, insert_with_string_values)
{
    auto query_string = m_query.set_table_name("strings")
                            .set_columns("col1", "col2")
                            .set_values("hello", "world")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "INSERT INTO strings ( col1, col2 ) VALUES ( \"hello\", \"world\" )");
}

TEST_F(table_insert_query_test, insert_with_numeric_values)
{
    auto query_string = m_query.set_table_name("numbers")
                            .set_columns("int_col", "float_col", "big_int")
                            .set_values(42, 3.14159, int64_t{ 1234567890123LL })
                            .get_query_string();
    EXPECT_TRUE(query_string.find("INSERT INTO numbers") != std::string::npos);
    EXPECT_TRUE(query_string.find("42") != std::string::npos);
    EXPECT_TRUE(query_string.find("3.14159") != std::string::npos);
    EXPECT_TRUE(query_string.find("1234567890123") != std::string::npos);
}

TEST_F(table_insert_query_test, insert_with_optional_null_value)
{
    std::optional<int> null_val     = std::nullopt;
    auto               query_string = m_query.set_table_name("nullable")
                            .set_columns("id", "optional_col")
                            .set_values(1, null_val)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "INSERT INTO nullable ( id, optional_col ) VALUES ( 1, NULL )");
}

TEST_F(table_insert_query_test, insert_with_optional_present_value)
{
    std::optional<int> present_val  = 42;
    auto               query_string = m_query.set_table_name("optional_test")
                            .set_columns("id", "value")
                            .set_values(1, present_val)
                            .get_query_string();
    EXPECT_EQ(query_string, "INSERT INTO optional_test ( id, value ) VALUES ( 1, 42 )");
}

TEST_F(table_insert_query_test, insert_with_mixed_optional_values)
{
    std::optional<int>    null_int       = std::nullopt;
    std::optional<double> present_double = 99.9;
    auto                  query_string   = m_query.set_table_name("mixed")
                            .set_columns("a", "b", "c", "d")
                            .set_values(1, null_int, "text", present_double)
                            .get_query_string();
    EXPECT_TRUE(query_string.find("INSERT INTO mixed") != std::string::npos);
    EXPECT_TRUE(query_string.find("( a, b, c, d )") != std::string::npos);
    EXPECT_TRUE(query_string.find("1,") != std::string::npos);
    EXPECT_TRUE(query_string.find("NULL") != std::string::npos);
    EXPECT_TRUE(query_string.find("\"text\"") != std::string::npos);
    EXPECT_TRUE(query_string.find("99.9") != std::string::npos);
}

TEST_F(table_insert_query_test, reuse_query_builder_resets_content)
{
    auto query1 = m_query.set_table_name("table1")
                      .set_columns("col1")
                      .set_values(100)
                      .get_query_string();
    EXPECT_EQ(query1, "INSERT INTO table1 ( col1 ) VALUES ( 100 )");

    auto query2 = m_query.set_table_name("table2")
                      .set_columns("col2")
                      .set_values(200)
                      .get_query_string();
    EXPECT_EQ(query2, "INSERT INTO table2 ( col2 ) VALUES ( 200 )");
}

TEST_F(table_insert_query_test, insert_with_many_columns)
{
    auto query_string = m_query.set_table_name("wide_table")
                            .set_columns("c1", "c2", "c3", "c4", "c5")
                            .set_values(1, 2, 3, 4, 5)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "INSERT INTO wide_table ( c1, c2, c3, c4, c5 ) VALUES ( 1, 2, 3, 4, 5 )");
}

TEST_F(table_insert_query_test, insert_with_negative_numbers)
{
    auto query_string = m_query.set_table_name("signed_numbers")
                            .set_columns("negative", "positive")
                            .set_values(-42, 42)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "INSERT INTO signed_numbers ( negative, positive ) VALUES ( -42, 42 )");
}

TEST_F(table_insert_query_test, insert_with_zero_values)
{
    auto query_string = m_query.set_table_name("zeros")
                            .set_columns("int_zero", "float_zero")
                            .set_values(0, 0.0)
                            .get_query_string();
    EXPECT_TRUE(query_string.find("INSERT INTO zeros") != std::string::npos);
    EXPECT_TRUE(query_string.find("( int_zero, float_zero )") != std::string::npos);
}

}  // namespace
