// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/ompt/decoder.hpp"

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace rocprofsys::domains::callback::ompt
{
namespace
{

constexpr std::array k_table{
    flag_bit{ .mask = 0x1, .value = "one" },
    flag_bit{ .mask = 0x2, .value = "two" },
    flag_bit{ .mask = 0x4, .value = "four" },
};

TEST(ompt_decoder_test, has_flag_checks_any_bit_of_mask)
{
    EXPECT_TRUE(has_flag(0x5, 0x4));
    EXPECT_FALSE(has_flag(0x5, 0x2));
}

TEST(ompt_decoder_test, has_flag_handles_bit_31_in_signed_flags)
{
    EXPECT_TRUE(has_flag(static_cast<int>(0x80000000U), 0x80000000U));
}

TEST(ompt_decoder_test, first_match_returns_earliest_table_entry)
{
    EXPECT_EQ(first_match(0x6, k_table), "two");
}

TEST(ompt_decoder_test, first_match_returns_nullopt_when_no_bit_set)
{
    EXPECT_FALSE(first_match(0x8, k_table).has_value());
}

TEST(ompt_decoder_test, all_matches_joins_values_in_table_order)
{
    EXPECT_EQ(all_matches(0x5, k_table), "one, four");
}

TEST(ompt_decoder_test, all_matches_reports_none_when_no_bit_set)
{
    EXPECT_EQ(all_matches(0x0, k_table), "none");
}

TEST(ompt_decoder_test, writer_numbers_arguments_and_skips_empty_match)
{
    auto                  args = function_args_t{};
    const flag_arg_writer writer{ args, 0x2, "t" };

    writer.first_match("absent", std::array{ flag_bit{ .mask = 0x8, .value = "x" } });
    writer.first_match("picked", k_table);
    writer.all_matches("all", k_table);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0].arg_number, 0U);
    EXPECT_EQ(args[0].arg_type, "t");
    EXPECT_EQ(args[0].arg_name, "picked");
    EXPECT_EQ(args[0].arg_value, "two");
    EXPECT_EQ(args[1].arg_number, 1U);
    EXPECT_EQ(args[1].arg_value, "two");
}

}  // namespace
}  // namespace rocprofsys::domains::callback::ompt
