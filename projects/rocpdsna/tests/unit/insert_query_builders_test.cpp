// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "insert/insert_query_builders.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <sstream>
#include <string>

namespace
{

using namespace rocpdsna::queries::insert;

class query_value_builder_test : public ::testing::Test
{
protected:
    std::stringstream m_ss;
};

TEST_F(query_value_builder_test, set_single_integer_value)
{
    query_value_builder builder(m_ss);
    builder.set_values(42);
    EXPECT_EQ(builder.get_query_string(), "( 42 )");
}

TEST_F(query_value_builder_test, set_single_double_value)
{
    query_value_builder builder(m_ss);
    builder.set_values(3.14);
    auto result = builder.get_query_string();
    EXPECT_TRUE(result.find("3.14") != std::string::npos);
    EXPECT_TRUE(result.find("(") != std::string::npos);
    EXPECT_TRUE(result.find(")") != std::string::npos);
}

TEST_F(query_value_builder_test, set_single_string_literal_value)
{
    query_value_builder builder(m_ss);
    builder.set_values("hello");
    EXPECT_EQ(builder.get_query_string(), "( \"hello\" )");
}

TEST_F(query_value_builder_test, set_multiple_integer_values)
{
    query_value_builder builder(m_ss);
    builder.set_values(1, 2, 3);
    EXPECT_EQ(builder.get_query_string(), "( 1, 2, 3 )");
}

TEST_F(query_value_builder_test, set_mixed_type_values)
{
    query_value_builder builder(m_ss);
    builder.set_values(100, "test", 2.5);
    auto result = builder.get_query_string();
    EXPECT_TRUE(result.find("100") != std::string::npos);
    EXPECT_TRUE(result.find("\"test\"") != std::string::npos);
    EXPECT_TRUE(result.find("2.5") != std::string::npos);
}

TEST_F(query_value_builder_test, set_optional_value_with_value)
{
    query_value_builder builder(m_ss);
    std::optional<int>  opt_val = 99;
    builder.set_values(opt_val);
    EXPECT_EQ(builder.get_query_string(), "( 99 )");
}

TEST_F(query_value_builder_test, set_optional_value_without_value)
{
    query_value_builder builder(m_ss);
    std::optional<int>  opt_val = std::nullopt;
    builder.set_values(opt_val);
    EXPECT_EQ(builder.get_query_string(), "( NULL )");
}

TEST_F(query_value_builder_test, set_mixed_values_with_optional)
{
    query_value_builder   builder(m_ss);
    std::optional<int>    opt_val    = std::nullopt;
    std::optional<double> opt_double = 1.5;
    builder.set_values(1, opt_val, "text", opt_double);
    auto result = builder.get_query_string();
    EXPECT_TRUE(result.find("1,") != std::string::npos);
    EXPECT_TRUE(result.find("NULL") != std::string::npos);
    EXPECT_TRUE(result.find("\"text\"") != std::string::npos);
    EXPECT_TRUE(result.find("1.5") != std::string::npos);
}

TEST_F(query_value_builder_test, set_int64_value)
{
    query_value_builder builder(m_ss);
    int64_t             large_val = 9223372036854775807LL;
    builder.set_values(large_val);
    EXPECT_EQ(builder.get_query_string(), "( 9223372036854775807 )");
}

TEST_F(query_value_builder_test, set_uint64_value)
{
    query_value_builder builder(m_ss);
    uint64_t            large_val = 18446744073709551615ULL;
    builder.set_values(large_val);
    EXPECT_EQ(builder.get_query_string(), "( 18446744073709551615 )");
}

class query_columns_builder_test : public ::testing::Test
{
protected:
    std::stringstream m_ss;
};

TEST_F(query_columns_builder_test, set_single_column)
{
    query_columns_builder builder(m_ss);
    auto&                 value_builder = builder.set_columns("id");
    value_builder.set_values(1);
    EXPECT_EQ(value_builder.get_query_string(), "( id ) VALUES ( 1 )");
}

TEST_F(query_columns_builder_test, set_multiple_columns)
{
    query_columns_builder builder(m_ss);
    auto&                 value_builder = builder.set_columns("id", "name", "value");
    value_builder.set_values(1, "test", 42);
    auto result = value_builder.get_query_string();
    EXPECT_TRUE(result.find("( id, name, value )") != std::string::npos);
    EXPECT_TRUE(result.find("VALUES") != std::string::npos);
    EXPECT_TRUE(result.find("( 1, \"test\", 42 )") != std::string::npos);
}

TEST_F(query_columns_builder_test, set_columns_with_optional_values)
{
    query_columns_builder builder(m_ss);
    std::optional<int>    opt_val       = std::nullopt;
    auto&                 value_builder = builder.set_columns("id", "optional_col");
    value_builder.set_values(1, opt_val);
    auto result = value_builder.get_query_string();
    EXPECT_TRUE(result.find("( id, optional_col )") != std::string::npos);
    EXPECT_TRUE(result.find("( 1, NULL )") != std::string::npos);
}

}  // namespace
