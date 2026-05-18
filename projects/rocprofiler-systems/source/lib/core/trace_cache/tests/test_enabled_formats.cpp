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
make_with(bool rocpd_enabled, bool perfetto_enabled, bool unified_memory_enabled = false,
          bool rocpd_parallel = true, bool perfetto_parallel = true,
          bool unified_memory_parallel = false)
{
    return enabled_formats_t{ std::vector<format_t>{
        { format_kind::rocpd, rocpd_parallel, rocpd_enabled, "rocpd" },
        { format_kind::perfetto, perfetto_parallel, perfetto_enabled, "perfetto" },
        { format_kind::unified_memory, unified_memory_parallel, unified_memory_enabled,
          "unified_memory" },
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
    EXPECT_EQ(make_with(false, true, true).names(), "perfetto, unified_memory");
    EXPECT_EQ(make_with(true, false).names(), "rocpd");
    EXPECT_EQ(make_with(false, true).names(), "perfetto");
    EXPECT_EQ(make_with(false, false, true).names(), "unified_memory");
}

TEST(enabled_formats_test, has_parallel_formats_true_when_any_parallel_enabled)
{
    EXPECT_TRUE(make_with(true, false).has_parallel_formats());
    EXPECT_TRUE(make_with(false, true).has_parallel_formats());
    EXPECT_TRUE(make_with(true, true).has_parallel_formats());
    EXPECT_FALSE(make_with(false, false).has_parallel_formats());
    EXPECT_FALSE(make_with(true, false, /*unified_memory_enabled=*/false,
                           /*rocpd_parallel=*/false)
                     .has_parallel_formats());
}

TEST(enabled_formats_test, has_sequential_formats_true_when_any_sequential_enabled)
{
    EXPECT_TRUE(make_with(false, true, /*unified_memory_enabled=*/false,
                          /*rocpd_parallel=*/true, /*perfetto_parallel=*/false)
                    .has_sequential_formats());
    EXPECT_TRUE(make_with(false, false, true).has_sequential_formats());
    EXPECT_TRUE(make_with(true, false, /*unified_memory_enabled=*/false,
                          /*rocpd_parallel=*/false)
                    .has_sequential_formats());
    EXPECT_FALSE(make_with(true, true).has_sequential_formats());
    EXPECT_FALSE(make_with(false, false).has_sequential_formats());
}

TEST(enabled_formats_test, get_parallel_formats_keeps_only_parallel_enabled)
{
    auto sub = make_with(true, true, /*unified_memory_enabled=*/false,
                         /*rocpd_parallel=*/true, /*perfetto_parallel=*/false)
                   .get_parallel_formats();

    EXPECT_EQ(sub.formats.size(), 1U);
    EXPECT_EQ(sub.formats.front().kind, format_kind::rocpd);
    EXPECT_TRUE(sub.formats.front().process_parallel);
    EXPECT_TRUE(sub.formats.front().enabled);
}

TEST(enabled_formats_test, get_parallel_formats_returns_both_when_all_parallel)
{
    auto sub = make_with(true, true).get_parallel_formats();

    EXPECT_EQ(sub.formats.size(), 2U);
    EXPECT_TRUE(sub.formats[0].process_parallel);
    EXPECT_TRUE(sub.formats[1].process_parallel);
}

TEST(enabled_formats_test, get_sequential_formats_keeps_only_sequential_enabled)
{
    auto sub = make_with(true, true, /*unified_memory_enabled=*/true,
                         /*rocpd_parallel=*/true, /*perfetto_parallel=*/false)
                   .get_sequential_formats();

    EXPECT_EQ(sub.formats.size(), 2U);
    EXPECT_EQ(sub.formats.front().kind, format_kind::perfetto);
    EXPECT_FALSE(sub.formats.front().process_parallel);
    EXPECT_TRUE(sub.formats.front().enabled);
    EXPECT_EQ(sub.formats.back().kind, format_kind::unified_memory);
    EXPECT_FALSE(sub.formats.back().process_parallel);
    EXPECT_TRUE(sub.formats.back().enabled);
}

TEST(enabled_formats_test, get_sequential_formats_empty_when_all_parallel)
{
    auto sub = make_with(true, true).get_sequential_formats();

    EXPECT_TRUE(sub.formats.empty());
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

TEST(enabled_formats_test, is_unified_memory_enabled_uses_format_kind)
{
    EXPECT_TRUE(make_with(false, false, true).is_unified_memory_enabled());
    EXPECT_FALSE(make_with(true, false, false).is_unified_memory_enabled());
}

}  // namespace rocprofsys::trace_cache::data
