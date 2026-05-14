// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/data_types.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace rocprofsys::trace_cache::data
{
namespace
{
enabled_formats_t
make_with(bool rocpd_enabled, bool perfetto_enabled)
{
    return enabled_formats_t{ std::vector<format_t>{
        { format_kind::rocpd, true, rocpd_enabled, "rocpd" },
        { format_kind::perfetto, false, perfetto_enabled, "perfetto" },
    } };
}
}  // namespace

TEST(enabled_formats_test, names_empty_when_no_formats_enabled)
{
    EXPECT_EQ(make_with(false, false).names(), "");
}

TEST(enabled_formats_test, names_joins_enabled_formats_with_comma)
{
    EXPECT_EQ(make_with(true, true).names(), "rocpd, perfetto");
    EXPECT_EQ(make_with(true, false).names(), "rocpd");
    EXPECT_EQ(make_with(false, true).names(), "perfetto");
}

TEST(enabled_formats_test, has_parallel_formats_true_when_rocpd_enabled)
{
    EXPECT_TRUE(make_with(true, false).has_parallel_formats());
    EXPECT_FALSE(make_with(false, true).has_parallel_formats());
    EXPECT_FALSE(make_with(false, false).has_parallel_formats());
}

TEST(enabled_formats_test, has_sequential_formats_true_when_perfetto_enabled)
{
    EXPECT_TRUE(make_with(false, true).has_sequential_formats());
    EXPECT_FALSE(make_with(true, false).has_sequential_formats());
    EXPECT_FALSE(make_with(false, false).has_sequential_formats());
}

TEST(enabled_formats_test, get_parallel_formats_filters_out_sequential)
{
    auto sub = make_with(true, true).get_parallel_formats();

    EXPECT_EQ(sub.formats.size(), 1U);
    EXPECT_EQ(sub.formats.front().kind, format_kind::rocpd);
    EXPECT_TRUE(sub.formats.front().process_parallel);
    EXPECT_TRUE(sub.formats.front().enabled);
}

TEST(enabled_formats_test, get_sequential_formats_filters_out_parallel)
{
    auto sub = make_with(true, true).get_sequential_formats();

    EXPECT_EQ(sub.formats.size(), 1U);
    EXPECT_EQ(sub.formats.front().kind, format_kind::perfetto);
    EXPECT_FALSE(sub.formats.front().process_parallel);
    EXPECT_TRUE(sub.formats.front().enabled);
}

TEST(enabled_formats_test, is_rocpd_enabled_uses_format_kind)
{
    EXPECT_TRUE(make_with(true, false).is_rocpd_enabled());
    EXPECT_FALSE(make_with(false, true).is_rocpd_enabled());
}

TEST(enabled_formats_test, is_perfetto_enabled_uses_format_kind)
{
    EXPECT_TRUE(make_with(false, true).is_perfetto_enabled());
    EXPECT_FALSE(make_with(true, false).is_perfetto_enabled());
}

}  // namespace rocprofsys::trace_cache::data
