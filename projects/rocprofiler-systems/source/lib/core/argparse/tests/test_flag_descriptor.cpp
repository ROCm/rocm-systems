// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/argparse/flag_descriptor.hpp"

#include <gtest/gtest.h>

namespace
{
using rocprofsys::argparse::count_spec;
using rocprofsys::argparse::flag_descriptor;
using rocprofsys::argparse::flag_group;
using rocprofsys::argparse::join_with;
using rocprofsys::argparse::value_kind;
using rocprofsys::common::update_mode;

constexpr int SENTINEL_UNSET = -1;
}  // namespace

TEST(CountSpec, ExactlyProducesExactOnly)
{
    constexpr auto spec = count_spec::exactly(3);
    EXPECT_EQ(spec.exact, 3);
    EXPECT_EQ(spec.min, SENTINEL_UNSET);
    EXPECT_EQ(spec.max, SENTINEL_UNSET);
}

TEST(CountSpec, RangeProducesMinAndMax)
{
    constexpr auto spec = count_spec::range(1, 5);
    EXPECT_EQ(spec.exact, SENTINEL_UNSET);
    EXPECT_EQ(spec.min, 1);
    EXPECT_EQ(spec.max, 5);
}

TEST(CountSpec, AtLeastProducesMinOnly)
{
    constexpr auto spec = count_spec::at_least(2);
    EXPECT_EQ(spec.exact, SENTINEL_UNSET);
    EXPECT_EQ(spec.min, 2);
    EXPECT_EQ(spec.max, SENTINEL_UNSET);
}

TEST(CountSpec, AtMostProducesMaxOnly)
{
    constexpr auto spec = count_spec::at_most(4);
    EXPECT_EQ(spec.exact, SENTINEL_UNSET);
    EXPECT_EQ(spec.min, SENTINEL_UNSET);
    EXPECT_EQ(spec.max, 4);
}

TEST(CountSpec, AnyLeavesAllUnset)
{
    constexpr auto spec = count_spec::any();
    EXPECT_EQ(spec.exact, SENTINEL_UNSET);
    EXPECT_EQ(spec.min, SENTINEL_UNSET);
    EXPECT_EQ(spec.max, SENTINEL_UNSET);
}

TEST(CountSpec, DefaultConstructedMatchesAny)
{
    constexpr count_spec defaulted{};
    constexpr auto       any_spec = count_spec::any();
    EXPECT_EQ(defaulted.exact, any_spec.exact);
    EXPECT_EQ(defaulted.min, any_spec.min);
    EXPECT_EQ(defaulted.max, any_spec.max);
}

TEST(FlagDescriptor, DefaultConstructedHasSafeDefaults)
{
    flag_descriptor descriptor{};

    EXPECT_TRUE(descriptor.names.empty());
    EXPECT_TRUE(descriptor.help.empty());
    EXPECT_TRUE(descriptor.dtype.empty());
    EXPECT_EQ(descriptor.kind, value_kind::flag);
    EXPECT_EQ(descriptor.join, join_with::none);
    EXPECT_TRUE(descriptor.env_vars.empty());
    EXPECT_EQ(descriptor.mode, update_mode::REPLACE);
    EXPECT_TRUE(descriptor.dedup_keys.empty());
    EXPECT_TRUE(descriptor.choices.empty());
    EXPECT_TRUE(descriptor.conflicts.empty());
    EXPECT_TRUE(descriptor.requires_.empty());
    EXPECT_EQ(descriptor.custom, nullptr);

    // Default count is "any" — no constraint.
    EXPECT_EQ(descriptor.count.exact, SENTINEL_UNSET);
    EXPECT_EQ(descriptor.count.min, SENTINEL_UNSET);
    EXPECT_EQ(descriptor.count.max, SENTINEL_UNSET);
}

TEST(FlagGroup, DefaultConstructedHasEmptyFlags)
{
    flag_group group{};
    EXPECT_TRUE(group.title.empty());
    EXPECT_TRUE(group.subtitle.empty());
    EXPECT_TRUE(group.flags.empty());
}
