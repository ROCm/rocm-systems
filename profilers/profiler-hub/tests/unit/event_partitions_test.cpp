// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_partitions.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace
{

using profiler_hub::id_range;
using profiler_hub::merge_event_partitions;
using profiler_hub::plan_id_ranges;
using profiler_hub::table_id_span;
using profiler_hub::reader_types::event_type_t;

constexpr auto region = event_type_t::region;
constexpr auto kernel = event_type_t::kernel_dispatch;

ph_event_t
event_at(uint64_t start, const char* name)
{
    return ph_event_t{ .start = start, .end = start + 1, .name = name };
}

std::vector<uint64_t>
starts_of(const std::vector<ph_event_t>& events)
{
    std::vector<uint64_t> starts;
    for(const auto& event : events)
    {
        starts.push_back(event.start);
    }
    return starts;
}

std::vector<std::string>
names_of(const std::vector<ph_event_t>& events)
{
    std::vector<std::string> names;
    for(const auto& event : events)
    {
        names.emplace_back(event.name);
    }
    return names;
}

TEST(plan_id_ranges_test, no_spans_give_no_ranges)
{
    EXPECT_TRUE(plan_id_ranges({}, 4).empty());
}

TEST(plan_id_ranges_test, one_part_covers_the_span_with_one_range)
{
    const std::vector<table_id_span> spans{ { region, 5, 20 } };

    const auto ranges = plan_id_ranges(spans, 1);

    ASSERT_EQ(ranges.size(), 1U);
    EXPECT_EQ(ranges[0].type, region);
    EXPECT_EQ(ranges[0].begin, 5U);
    EXPECT_EQ(ranges[0].end, 21U);
}

TEST(plan_id_ranges_test, a_span_is_cut_into_equal_ranges)
{
    const std::vector<table_id_span> spans{ { region, 1, 8 } };

    const auto ranges = plan_id_ranges(spans, 4);

    ASSERT_EQ(ranges.size(), 4U);
    EXPECT_EQ(ranges[0].begin, 1U);
    EXPECT_EQ(ranges[0].end, 3U);
    EXPECT_EQ(ranges[1].begin, 3U);
    EXPECT_EQ(ranges[1].end, 5U);
    EXPECT_EQ(ranges[2].begin, 5U);
    EXPECT_EQ(ranges[2].end, 7U);
    EXPECT_EQ(ranges[3].begin, 7U);
    EXPECT_EQ(ranges[3].end, 9U);
}

TEST(plan_id_ranges_test, more_parts_than_ids_give_one_range_per_id)
{
    const std::vector<table_id_span> spans{ { region, 1, 2 } };

    const auto ranges = plan_id_ranges(spans, 8);

    ASSERT_EQ(ranges.size(), 2U);
    EXPECT_EQ(ranges[0].end - ranges[0].begin, 1U);
    EXPECT_EQ(ranges[1].end - ranges[1].begin, 1U);
}

TEST(plan_id_ranges_test, zero_parts_still_read_every_table_once)
{
    const std::vector<table_id_span> spans{ { region, 1, 10 }, { kernel, 1, 4 } };

    const auto ranges = plan_id_ranges(spans, 0);

    ASSERT_EQ(ranges.size(), 2U);
    EXPECT_EQ(ranges[0].type, region);
    EXPECT_EQ(ranges[1].type, kernel);
}

TEST(plan_id_ranges_test, tables_get_ranges_in_proportion_to_their_span)
{
    const std::vector<table_id_span> spans{ { region, 1, 9 }, { kernel, 1, 3 } };

    const auto ranges = plan_id_ranges(spans, 4);

    std::map<event_type_t, size_t> per_table;
    for(const auto& range : ranges)
    {
        ++per_table[range.type];
    }
    EXPECT_EQ(per_table[region], 3U);
    EXPECT_EQ(per_table[kernel], 1U);
}

TEST(plan_id_ranges_test, ranges_cover_every_span_exactly)
{
    const std::vector<std::vector<table_id_span>> cases{
        { { region, 1, 1 } },
        { { region, 0, 99 }, { kernel, 7, 8 } },
        { { region, 3, 1000003 },
          { kernel, 1, 17 },
          { event_type_t::memory_copy, 5, 5 } },
    };

    for(const auto& spans : cases)
    {
        for(const size_t parts : { 1U, 2U, 3U, 7U, 8U, 64U })
        {
            const auto ranges = plan_id_ranges(spans, parts);

            size_t next       = 0;
            size_t span_index = 0;
            for(const auto& range : ranges)
            {
                while(span_index < spans.size() && spans[span_index].type != range.type)
                {
                    ++span_index;
                }
                ASSERT_LT(span_index, spans.size());
                const auto& span = spans[span_index];
                if(range.begin == span.first)
                {
                    next = span.first;
                }
                EXPECT_EQ(range.begin, next) << "gap or overlap, parts=" << parts;
                EXPECT_LT(range.begin, range.end);
                next = range.end;
                EXPECT_LE(range.end, span.last + 1);
            }
        }
    }
}

TEST(plan_id_ranges_test, the_last_range_of_each_table_ends_after_its_last_id)
{
    const std::vector<table_id_span> spans{ { region, 1, 10 }, { kernel, 4, 13 } };

    const auto ranges = plan_id_ranges(spans, 5);

    size_t region_end = 0;
    size_t kernel_end = 0;
    for(const auto& range : ranges)
    {
        (range.type == region ? region_end : kernel_end) = range.end;
    }
    EXPECT_EQ(region_end, 11U);
    EXPECT_EQ(kernel_end, 14U);
}

TEST(plan_id_ranges_test, a_span_with_last_before_first_is_ignored)
{
    const std::vector<table_id_span> spans{ { region, 9, 3 }, { kernel, 1, 4 } };

    const auto ranges = plan_id_ranges(spans, 2);

    ASSERT_FALSE(ranges.empty());
    for(const auto& range : ranges)
    {
        EXPECT_EQ(range.type, kernel);
    }
}

TEST(merge_event_partitions_test, nothing_to_merge_gives_an_empty_list)
{
    std::vector<std::vector<ph_event_t>> outputs;

    EXPECT_TRUE(merge_event_partitions({}, outputs).empty());
}

TEST(merge_event_partitions_test, partitions_of_one_table_are_merged_by_start)
{
    const std::vector<id_range>          ranges{ { region, 0, 2 }, { region, 2, 4 } };
    std::vector<std::vector<ph_event_t>> outputs{
        { event_at(30, "c"), event_at(10, "a") },
        { event_at(20, "b"), event_at(40, "d") },
    };

    const auto merged = merge_event_partitions(ranges, outputs);

    EXPECT_THAT(starts_of(merged), ::testing::ElementsAre(10, 20, 30, 40));
    EXPECT_THAT(names_of(merged), ::testing::ElementsAre("a", "b", "c", "d"));
}

TEST(merge_event_partitions_test, tables_are_interleaved_by_start)
{
    const std::vector<id_range>          ranges{ { region, 0, 2 }, { kernel, 0, 2 } };
    std::vector<std::vector<ph_event_t>> outputs{
        { event_at(10, "r1"), event_at(30, "r2") },
        { event_at(20, "k1"), event_at(40, "k2") },
    };

    const auto merged = merge_event_partitions(ranges, outputs);

    EXPECT_THAT(names_of(merged), ::testing::ElementsAre("r1", "k1", "r2", "k2"));
}

TEST(merge_event_partitions_test, equal_starts_keep_the_order_of_the_tables)
{
    const std::vector<id_range>          ranges{ { region, 0, 1 }, { kernel, 0, 1 } };
    std::vector<std::vector<ph_event_t>> outputs{
        { event_at(10, "region") },
        { event_at(10, "kernel") },
    };

    const auto merged = merge_event_partitions(ranges, outputs);

    EXPECT_THAT(names_of(merged), ::testing::ElementsAre("region", "kernel"));
}

TEST(merge_event_partitions_test, the_inputs_are_emptied)
{
    const std::vector<id_range>          ranges{ { region, 0, 1 }, { region, 1, 2 } };
    std::vector<std::vector<ph_event_t>> outputs{ { event_at(1, "a") },
                                                  { event_at(2, "b") } };

    const auto merged = merge_event_partitions(ranges, outputs);

    EXPECT_EQ(merged.size(), 2U);
    EXPECT_TRUE(outputs[0].empty());
    EXPECT_TRUE(outputs[1].empty());
}

}  // namespace
