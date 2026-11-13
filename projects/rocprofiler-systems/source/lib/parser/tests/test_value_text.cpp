// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr std::int64_t k_small_integer = 42;
constexpr std::int64_t k_limit         = 10;
constexpr std::int64_t k_thousand      = 1000;
constexpr double       k_whole_number  = 1000.0;
constexpr double       k_fraction      = 0.25;
constexpr double       k_thousandth    = 0.001;

schema_param
make_param(param_type type)
{
    auto param = schema_param{};
    param.id   = "param";
    param.type = type;
    return param;
}

schema_param
make_list()
{
    auto param    = make_param(param_type::enumeration_list);
    param.choices = { choice{ .value = "alpha" }, choice{ .value = "beta" } };
    return param;
}
}  // namespace

TEST(parser_value_text, parse_boolean_reads_words_and_integers)
{
    EXPECT_EQ(parse_boolean("Yes"), true);
    EXPECT_EQ(parse_boolean("off"), false);
    EXPECT_EQ(parse_boolean("+2"), true);
    EXPECT_EQ(parse_boolean("0"), false);
    EXPECT_EQ(parse_boolean(""), std::nullopt);
    EXPECT_EQ(parse_boolean("maybe"), std::nullopt);
}

TEST(parser_value_text, parse_integer_reads_the_whole_string)
{
    constexpr auto k_largest = std::numeric_limits<std::int64_t>::max();
    EXPECT_EQ(parse_integer("-42"), -k_small_integer);
    EXPECT_EQ(parse_integer(std::to_string(k_largest)), k_largest);
    for(const auto* text : { "", "+1", "1.0", " 1", "1 ", "0x10", "9223372036854775808" })
    {
        EXPECT_EQ(parse_integer(text), std::nullopt) << text;
    }
}

TEST(parser_value_text, parse_number_reads_finite_numbers)
{
    EXPECT_EQ(parse_number("0.25"), k_fraction);
    EXPECT_EQ(parse_number("-1e3"), -k_whole_number);
    for(const auto* text :
        { "", "nan", "inf", "-infinity", "1e999", " 1", "1.5x", "0x1p3" })
    {
        EXPECT_EQ(parse_number(text), std::nullopt) << text;
    }
}

TEST(parser_value_text, exact_integer_bound_needs_a_whole_number_in_range)
{
    constexpr auto k_lowest  = std::numeric_limits<std::int64_t>::min();
    constexpr auto k_largest = std::numeric_limits<std::int64_t>::max();
    EXPECT_EQ(exact_integer_bound(k_whole_number), k_thousand);
    EXPECT_EQ(exact_integer_bound(-1.0), -1);
    EXPECT_EQ(exact_integer_bound(static_cast<double>(k_lowest)), k_lowest);
    EXPECT_EQ(exact_integer_bound(k_fraction), std::nullopt);
    // As a double, the std::int64_t maximum is 2^63, one past the range.
    EXPECT_EQ(exact_integer_bound(static_cast<double>(k_largest)), std::nullopt);
    EXPECT_EQ(exact_integer_bound(std::numeric_limits<double>::infinity()), std::nullopt);
    EXPECT_EQ(exact_integer_bound(std::numeric_limits<double>::quiet_NaN()),
              std::nullopt);
}

TEST(parser_value_text, parse_value_returns_the_typed_value)
{
    EXPECT_EQ(parse_value(make_param(param_type::boolean), "on"), config_value_t{ true });
    EXPECT_EQ(parse_value(make_param(param_type::integer), "42"),
              config_value_t{ k_small_integer });
    EXPECT_EQ(parse_value(make_param(param_type::number), "0.25"),
              config_value_t{ k_fraction });
    EXPECT_EQ(parse_value(make_param(param_type::path), "/tmp"),
              config_value_t{ std::string{ "/tmp" } });
    EXPECT_EQ(parse_value(make_list(), " alpha , beta "),
              (config_value_t{ std::vector<std::string>{ "alpha", "beta" } }));
    EXPECT_EQ(parse_value(make_param(param_type::integer), "seven"), std::nullopt);
    EXPECT_EQ(parse_value(make_list(), "alpha,gamma"), std::nullopt);
}

TEST(parser_value_text, accepts_value_checks_the_type_and_the_rules)
{
    auto bounded = make_param(param_type::integer);
    bounded.max  = static_cast<double>(k_limit);
    EXPECT_TRUE(accepts_value(bounded, k_limit));
    EXPECT_FALSE(accepts_value(bounded, k_limit + 1));
    EXPECT_FALSE(accepts_value(bounded, static_cast<double>(k_limit)));
    EXPECT_FALSE(accepts_value(make_param(param_type::path), std::string{}));
    EXPECT_TRUE(accepts_value(make_list(), std::vector<std::string>{}));
    EXPECT_FALSE(accepts_value(make_list(), std::vector<std::string>{ "alpha", "" }));
}

TEST(parser_value_text, format_value_writes_canonical_text)
{
    EXPECT_EQ(format_value(true), "true");
    EXPECT_EQ(format_value(-k_small_integer), "-42");
    EXPECT_EQ(format_value(k_whole_number), "1000");
    EXPECT_EQ(format_value(k_fraction), "0.25");
    EXPECT_EQ(format_value(std::string{ "text" }), "text");
    EXPECT_EQ(format_value(std::vector<std::string>{ "alpha", "beta" }), "alpha,beta");
}

TEST(parser_value_text, a_formatted_number_parses_back_to_the_same_value)
{
    const auto param = make_param(param_type::number);
    EXPECT_EQ(parse_value(param, "1e-3"), config_value_t{ k_thousandth });
    EXPECT_EQ(format_value(k_thousandth), "0.001");
    EXPECT_EQ(parse_value(param, format_value(k_thousandth)),
              config_value_t{ k_thousandth });
}
