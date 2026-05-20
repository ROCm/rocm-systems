// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// parsed_values is the engine-agnostic accessor handed to custom actions.
// Tests construct a real parser_t, register a synthetic flag, parse argv,
// then verify each get<T> specialization + exists + set_use_color through
// the same path the production interpreter takes.

#include "core/argparse/detail/parser_engine.hpp"
#include "core/argparse/parsed_values.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <deque>
#include <set>
#include <string>
#include <vector>

namespace
{
using rocprofsys::argparse::parsed_values;
using rocprofsys::argparse::parser_t;

class ParsedValuesTest : public ::testing::Test
{
protected:
    parser_t parser{ "test" };

    void SetUp() override { parser.enable_help(); }

    void parse(std::vector<std::string> args)
    {
        args.insert(args.begin(), "test");
        const auto err = parser.parse(args, 0);
        ASSERT_FALSE(err) << err.what();
    }
};

}  // namespace

TEST_F(ParsedValuesTest, GetBoolFlag)
{
    parser.add_argument({ std::string{ "--flag" } }, std::string{ "f" }).max_count(1);
    parse({ "--flag" });

    parsed_values values{ parser };
    EXPECT_TRUE(values.get<bool>("flag"));
}

TEST_F(ParsedValuesTest, GetInt)
{
    parser.add_argument({ std::string{ "--n" } }, std::string{ "n" })
        .count(1)
        .dtype("int");
    parse({ "--n", "7" });

    parsed_values values{ parser };
    EXPECT_EQ(values.get<int>("n"), 7);
}

TEST_F(ParsedValuesTest, GetInt64)
{
    parser.add_argument({ std::string{ "--big" } }, std::string{ "big int" })
        .count(1)
        .dtype("int64");
    parse({ "--big", "1234567890123" });

    parsed_values values{ parser };
    EXPECT_EQ(values.get<std::int64_t>("big"), std::int64_t{ 1234567890123 });
}

TEST_F(ParsedValuesTest, GetDouble)
{
    parser.add_argument({ std::string{ "--d" } }, std::string{ "d" })
        .count(1)
        .dtype("float");
    parse({ "--d", "3.25" });

    parsed_values values{ parser };
    EXPECT_DOUBLE_EQ(values.get<double>("d"), 3.25);
}

TEST_F(ParsedValuesTest, GetString)
{
    parser.add_argument({ std::string{ "--name" } }, std::string{ "name" })
        .count(1)
        .dtype("string");
    parse({ "--name", "marjan" });

    parsed_values values{ parser };
    EXPECT_EQ(values.get<std::string>("name"), "marjan");
}

TEST_F(ParsedValuesTest, GetStringSet)
{
    parser.add_argument({ std::string{ "--tags" } }, std::string{ "tags" })
        .min_count(1)
        .dtype("strings");
    parse({ "--tags", "a", "b", "a" });  // duplicates collapsed by set

    parsed_values values{ parser };
    auto          result = values.get<std::set<std::string>>("tags");
    EXPECT_EQ(result.size(), 2u);
    EXPECT_TRUE(result.count("a") > 0);
    EXPECT_TRUE(result.count("b") > 0);
}

TEST_F(ParsedValuesTest, GetStringVector)
{
    parser.add_argument({ std::string{ "--list" } }, std::string{ "list" })
        .min_count(1)
        .dtype("strings");
    parse({ "--list", "x", "y", "z" });

    parsed_values values{ parser };
    auto          result = values.get<std::vector<std::string>>("list");
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], "x");
    EXPECT_EQ(result[2], "z");
}

TEST_F(ParsedValuesTest, GetStringDeque)
{
    parser.add_argument({ std::string{ "--queue" } }, std::string{ "queue" })
        .min_count(1)
        .dtype("strings");
    parse({ "--queue", "first", "second" });

    parsed_values values{ parser };
    auto          result = values.get<std::deque<std::string>>("queue");
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.front(), "first");
    EXPECT_EQ(result.back(), "second");
}

TEST_F(ParsedValuesTest, GetInt64Vector)
{
    parser.add_argument({ std::string{ "--ids" } }, std::string{ "ids" })
        .min_count(1)
        .dtype("int and/or range");
    parse({ "--ids", "10", "20", "30" });

    parsed_values values{ parser };
    auto          result = values.get<std::vector<std::int64_t>>("ids");
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[2], 30);
}

TEST_F(ParsedValuesTest, ExistsReturnsTrueForProvidedFlag)
{
    parser.add_argument({ std::string{ "--maybe" } }, std::string{ "maybe" })
        .max_count(1);
    parse({ "--maybe" });

    parsed_values values{ parser };
    EXPECT_TRUE(values.exists("maybe"));
}

TEST_F(ParsedValuesTest, ExistsReturnsFalseForOmittedFlag)
{
    parser.add_argument({ std::string{ "--maybe" } }, std::string{ "maybe" })
        .max_count(1);
    parse({});  // do not provide --maybe

    parsed_values values{ parser };
    EXPECT_FALSE(values.exists("maybe"));
}

TEST_F(ParsedValuesTest, SetUseColorPropagatesToEngine)
{
    parsed_values values{ parser };
    values.set_use_color(false);  // smoke: must not throw
    values.set_use_color(true);
    SUCCEED();
}
