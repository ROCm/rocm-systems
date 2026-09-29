// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/ompt/decoder.hpp"

#include <gtest/gtest.h>

#include <array>

namespace rocprofsys::domains::callback::ompt
{
namespace
{

enum class test_flag_t : int
{
    a = 0x1,
    b = 0x2,
    c = 0x4,
};

// ─── has_flag / to_underlying ────────────────────────────────────────────────

TEST(ompt_flag_utils_test, has_flag_detects_enum_bit)
{
    EXPECT_TRUE(has_flag(0x3, test_flag_t::a));
    EXPECT_TRUE(has_flag(0x3, test_flag_t::b));
    EXPECT_FALSE(has_flag(0x3, test_flag_t::c));
}

TEST(ompt_flag_utils_test, has_flag_detects_raw_int_mask)
{
    EXPECT_TRUE(has_flag(0x5, 0x4));
    EXPECT_FALSE(has_flag(0x5, 0x2));
}

TEST(ompt_flag_utils_test, to_underlying_passes_through_non_enum)
{
    EXPECT_EQ(to_underlying(42), 42);
}

TEST(ompt_flag_utils_test, to_underlying_converts_enum_to_its_underlying_value)
{
    EXPECT_EQ(to_underlying(test_flag_t::b), 0x2);
}

// ─── flag_representation::decode ─────────────────────────────────────────────

using rules::flag_decode_mode;
using rules::no_match_behavior;
using test_rule = rules::flag_rule<int>;

TEST(flag_representation_test, first_match_emits_matching_rule)
{
    static constexpr std::array<test_rule, 2> k_table{
        test_rule{ .mask  = 0x1,
                   .type  = "t",
                   .key   = "k",
                   .value = "one",
                   .mode  = flag_decode_mode::first_match },
        test_rule{ .mask  = 0x2,
                   .type  = "t",
                   .key   = "k",
                   .value = "two",
                   .mode  = flag_decode_mode::first_match },
    };

    const auto decoded = flag_representation<int>{ 0x2, k_table }.decode();

    ASSERT_EQ(decoded.size(), 1U);
    EXPECT_EQ(decoded[0].type, "t");
    EXPECT_EQ(decoded[0].key, "k");
    EXPECT_EQ(decoded[0].value, "two");
}

TEST(flag_representation_test, first_match_prefers_earlier_rule_when_both_match)
{
    static constexpr std::array<test_rule, 2> k_table{
        test_rule{ .mask  = 0x1,
                   .type  = "t",
                   .key   = "k",
                   .value = "one",
                   .mode  = flag_decode_mode::first_match },
        test_rule{ .mask  = 0x2,
                   .type  = "t",
                   .key   = "k",
                   .value = "two",
                   .mode  = flag_decode_mode::first_match },
    };

    const auto decoded = flag_representation<int>{ 0x3, k_table }.decode();

    ASSERT_EQ(decoded.size(), 1U);
    EXPECT_EQ(decoded[0].value, "one");
}

TEST(flag_representation_test, first_match_emits_nothing_when_no_rule_matches)
{
    static constexpr std::array<test_rule, 1> k_table{
        test_rule{ .mask  = 0x1,
                   .type  = "t",
                   .key   = "k",
                   .value = "one",
                   .mode  = flag_decode_mode::first_match },
    };

    EXPECT_TRUE(flag_representation<int>(0x0, k_table).decode().empty());
}

TEST(flag_representation_test, all_matches_joins_matched_values_with_comma)
{
    static constexpr std::array<test_rule, 3> k_table{
        test_rule{ .mask  = 0x1,
                   .type  = "t",
                   .key   = "props",
                   .value = "a",
                   .mode  = flag_decode_mode::all_matches },
        test_rule{ .mask  = 0x2,
                   .type  = "t",
                   .key   = "props",
                   .value = "b",
                   .mode  = flag_decode_mode::all_matches },
        test_rule{ .mask  = 0x4,
                   .type  = "t",
                   .key   = "props",
                   .value = "c",
                   .mode  = flag_decode_mode::all_matches },
    };

    const auto decoded = flag_representation<int>{ 0x5, k_table }.decode();

    ASSERT_EQ(decoded.size(), 1U);
    EXPECT_EQ(decoded[0].value, "a, c");
}

TEST(flag_representation_test, all_matches_omits_group_when_no_match_and_behavior_is_omit)
{
    static constexpr std::array<test_rule, 1> k_table{
        test_rule{ .mask        = 0x1,
                   .type        = "t",
                   .key         = "props",
                   .value       = "a",
                   .mode        = flag_decode_mode::all_matches,
                   .on_no_match = no_match_behavior::omit },
    };

    EXPECT_TRUE(flag_representation<int>(0x0, k_table).decode().empty());
}

TEST(flag_representation_test,
     all_matches_emits_none_when_no_match_and_behavior_is_emit_none)
{
    static constexpr std::array<test_rule, 1> k_table{
        test_rule{ .mask        = 0x1,
                   .type        = "t",
                   .key         = "props",
                   .value       = "a",
                   .mode        = flag_decode_mode::all_matches,
                   .on_no_match = no_match_behavior::emit_none },
    };

    const auto decoded = flag_representation<int>{ 0x0, k_table }.decode();

    ASSERT_EQ(decoded.size(), 1U);
    EXPECT_EQ(decoded[0].value, "none");
}

TEST(flag_representation_test, decodes_multiple_independent_groups_in_order)
{
    static constexpr std::array<test_rule, 3> k_table{
        test_rule{ .mask  = 0x1,
                   .type  = "t",
                   .key   = "kind",
                   .value = "x",
                   .mode  = flag_decode_mode::first_match },
        test_rule{ .mask  = 0x2,
                   .type  = "t",
                   .key   = "props",
                   .value = "y",
                   .mode  = flag_decode_mode::all_matches },
        test_rule{ .mask  = 0x4,
                   .type  = "t",
                   .key   = "props",
                   .value = "z",
                   .mode  = flag_decode_mode::all_matches },
    };

    const auto decoded = flag_representation<int>{ 0x1 | 0x4, k_table }.decode();

    ASSERT_EQ(decoded.size(), 2U);
    EXPECT_EQ(decoded[0].key, "kind");
    EXPECT_EQ(decoded[0].value, "x");
    EXPECT_EQ(decoded[1].key, "props");
    EXPECT_EQ(decoded[1].value, "z");
}

}  // namespace
}  // namespace rocprofsys::domains::callback::ompt
