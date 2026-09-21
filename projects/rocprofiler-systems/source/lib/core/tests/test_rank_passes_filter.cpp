// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/config.hpp"

#include <cstdint>
#include <optional>
#include <string>

#if ROCPROFSYS_MPI_OR_MPI_HEADERS_ENABLED

namespace
{
using rocprofsys::config::output_filtering::rank_passes_filter;

constexpr std::optional<std::uint64_t> no_value = std::nullopt;
}  // namespace

// --- keyword shortcuts (evaluated before rank/world-size are consulted) ---

TEST(rank_passes_filter, empty_string_enables_all)
{
    EXPECT_TRUE(rank_passes_filter(2, 4, ""));
}

TEST(rank_passes_filter, all_keyword_enables)
{
    EXPECT_TRUE(rank_passes_filter(2, 4, "all"));
}

TEST(rank_passes_filter, none_keyword_disables)
{
    EXPECT_FALSE(rank_passes_filter(2, 4, "none"));
}

TEST(rank_passes_filter, keyword_is_case_insensitive)
{
    EXPECT_TRUE(rank_passes_filter(2, 4, "ALL"));
    EXPECT_FALSE(rank_passes_filter(2, 4, "None"));
}

// --- missing current rank -> filtering disabled (rank produces output) ---

TEST(rank_passes_filter, missing_current_rank_disables_filter)
{
    EXPECT_TRUE(rank_passes_filter(no_value, 4, "0"));
}

// --- basic membership, world size known ---

TEST(rank_passes_filter, current_rank_in_filter)
{
    EXPECT_TRUE(rank_passes_filter(2, 4, "0-3"));
}

TEST(rank_passes_filter, current_rank_not_in_filter)
{
    EXPECT_FALSE(rank_passes_filter(2, 4, "0,1"));
}

TEST(rank_passes_filter, current_rank_explicitly_listed)
{
    EXPECT_TRUE(rank_passes_filter(3, 8, "1,3,5"));
}

// --- world-size validation ---

TEST(rank_passes_filter, all_filter_ranks_out_of_range_disables_filter)
{
    // every requested rank >= world size -> none survive -> filter disabled
    EXPECT_TRUE(rank_passes_filter(1, 3, "5,6"));
}

TEST(rank_passes_filter, out_of_range_ranks_pruned_in_range_kept)
{
    // 5 is pruned (>= world size 4); current rank 2 is in {2} -> enabled
    EXPECT_TRUE(rank_passes_filter(2, 4, "2,5"));
    // current rank 1 is not in surviving {2} -> disabled
    EXPECT_FALSE(rank_passes_filter(1, 4, "2,5"));
}

TEST(rank_passes_filter, current_rank_out_of_range_disables_filter)
{
    // current rank 10 >= world size 3; filter "0" would otherwise exclude it
    EXPECT_TRUE(rank_passes_filter(10, 3, "0"));
}

TEST(rank_passes_filter, world_size_zero_disables_filter)
{
    EXPECT_TRUE(rank_passes_filter(0, 0, "0"));
}

// --- world size unknown -> no validation, plain membership ---

TEST(rank_passes_filter, no_world_size_skips_validation_member)
{
    EXPECT_TRUE(rank_passes_filter(100, no_value, "0-200"));
}

TEST(rank_passes_filter, no_world_size_skips_validation_non_member)
{
    EXPECT_FALSE(rank_passes_filter(5, no_value, "0-3"));
}

// --- invalid filter string -> no valid ranks -> filter disabled ---

TEST(rank_passes_filter, all_invalid_tokens_disables_filter)
{
    EXPECT_TRUE(rank_passes_filter(2, 4, "garbage"));
}

TEST(rank_passes_filter, invalid_tokens_pruned_valid_membership_kept)
{
    // "garbage" and reversed "3-1" dropped; "2" survives, current rank 2 -> enabled
    EXPECT_TRUE(rank_passes_filter(2, 8, "garbage,3-1,2"));
    // same filter, current rank 5 not in {2} -> disabled
    EXPECT_FALSE(rank_passes_filter(5, 8, "garbage,3-1,2"));
}

#endif
