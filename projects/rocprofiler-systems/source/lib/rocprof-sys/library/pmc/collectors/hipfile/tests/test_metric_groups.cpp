// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/hipfile/types.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>

namespace rocprofsys::pmc::collectors::hipfile::testing
{
namespace
{
constexpr std::array k_all_groups{ "bytes",  "ops",       "fastpath", "fallback",
                                   "errors", "unaligned", "bandwidth" };

TEST(HipFileMetricGroups, each_group_covers_exactly_one_read_and_one_write)
{
    for(const auto* group : k_all_groups)
    {
        EXPECT_EQ(std::popcount(metric_group_mask(group)), 2) << group;
    }
}

TEST(HipFileMetricGroups, groups_partition_the_metric_table)
{
    // Every track reachable through exactly one group: no metric is orphaned (and so
    // unselectable) and none is claimed twice.
    std::uint32_t combined = 0;
    int           bits     = 0;
    for(const auto* group : k_all_groups)
    {
        combined |= metric_group_mask(group);
        bits += std::popcount(metric_group_mask(group));
    }

    EXPECT_EQ(combined, ALL_HIPFILE_METRICS);
    EXPECT_EQ(bits, static_cast<int>(METRIC_TABLE.size()));
}

TEST(HipFileMetricGroups, unknown_group_selects_nothing)
{
    EXPECT_EQ(metric_group_mask(""), 0U);
    EXPECT_EQ(metric_group_mask("nonsense"), 0U);
    // The per-track spellings were replaced by group names, so they must not resolve.
    EXPECT_EQ(metric_group_mask("read_bytes"), 0U);
}

}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
