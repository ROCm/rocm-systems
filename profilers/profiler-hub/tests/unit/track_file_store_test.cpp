// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "depth_tracker.hpp"
#include "ph_future.hpp"
#include "sql_cancel.hpp"
#include "track_file_store.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{

using profiler_hub::depth_tracker;
using profiler_hub::track_file_store;

struct cache_fixture_t
{
    std::filesystem::path root;
    std::filesystem::path config;
    std::filesystem::path trace;

    cache_fixture_t()
    {
        root = std::filesystem::temp_directory_path() / "profiler-hub-track-store-test";
        std::filesystem::remove_all(root);
        config = root / "config";
        std::filesystem::create_directories(config);
        trace = root / "trace.rpd";
        std::ofstream{ trace } << "trace-bytes";
    }

    ~cache_fixture_t() { std::filesystem::remove_all(root); }
};

ph_event_t
make_event(uint64_t id, uint64_t start, uint64_t end, const char* name)
{
    return ph_event_t{ .id            = id,
                       .start         = start,
                       .end           = end,
                       .name          = name,
                       .type          = PH_EVENT_TYPE_KERNEL_DISPATCH,
                       .nesting_depth = 0,
                       .track_id      = 7 };
}

void
assign_depths(std::vector<ph_event_t>& events)
{
    depth_tracker tracker;
    for(ph_event_t& event : events)
    {
        event.nesting_depth = static_cast<uint32_t>(tracker.add(event.start, event.end));
    }
}

}  // namespace

TEST(track_file_store_test, a_window_keeps_a_long_event_from_an_earlier_checkpoint)
{
    cache_fixture_t         fixture;
    std::vector<ph_event_t> events;
    events.push_back(make_event(1, 0, 10000, "long"));
    events.push_back(make_event(2, 10, 20, "early"));
    events.push_back(make_event(3, 30, 40, "middle"));
    events.push_back(make_event(4, 50, 60, "later"));
    assign_depths(events);

    const std::vector<std::string> sources{ fixture.trace.string() };
    {
        track_file_store store{ 2 };
        ASSERT_TRUE(store.open(fixture.config.string(), sources, 1, 2, 0, 0));
        EXPECT_FALSE(store.ready());
        ASSERT_TRUE(store.begin_write());
        store.set_histogram_grid(0, 1000, 10);
        store.write_events(7, 0, 7, events);
        store.write_samples(
            8,
            0,
            8,
            { ph_sample_t{ 0, 1.0 }, ph_sample_t{ 100, 2.0 }, ph_sample_t{ 200, 3.0 } },
            400);
        ASSERT_TRUE(store.finish_write());
    }

    track_file_store loaded;
    ASSERT_TRUE(loaded.open(fixture.config.string(), sources, 1, 2, 0, 0));
    ASSERT_TRUE(loaded.ready());
    EXPECT_EQ(loaded.nesting_depth(7), 2U);
    EXPECT_EQ(loaded.record_count(7), 4U);

    const ph_event_list_t slice = loaded.events(7, 5000, 6000);
    ASSERT_EQ(slice.list_size, 1U);
    EXPECT_EQ(slice.events[0].id, 1U);
    EXPECT_EQ(slice.events[0].nesting_depth, 1U);
    EXPECT_STREQ(slice.events[0].name, "long");

    const ph_event_list_t overlap = loaded.events(7, 15, 16);
    ASSERT_EQ(overlap.list_size, 2U);
    EXPECT_EQ(overlap.events[0].id, 1U);
    EXPECT_EQ(overlap.events[1].id, 2U);
    EXPECT_EQ(overlap.events[1].nesting_depth, 2U);
    EXPECT_EQ(loaded.nesting_depth_of(0, 7, 2, PH_EVENT_TYPE_KERNEL_DISPATCH), 2U);

    const ph_sample_list_t tail = loaded.samples(8, 250, 300);
    ASSERT_EQ(tail.list_size, 2U);
    EXPECT_EQ(tail.samples[0].timestamp, 200U);
    EXPECT_EQ(tail.samples[1].timestamp, 400U);
    EXPECT_DOUBLE_EQ(tail.samples[1].value, 3.0);

    const auto histogram = loaded.histogram(0, 1000, 10);
    ASSERT_EQ(histogram.tracks.size(), 2U);
    EXPECT_EQ(histogram.tracks[0].track_id, 7U);
    EXPECT_EQ(histogram.tracks[0].buckets.size(), 10U);

    loaded.release_events(slice.events);
    loaded.release_events(overlap.events);
    loaded.release_samples(tail.samples);

    auto future = ph_future::create(nullptr, nullptr);
    ASSERT_EQ(future->cancel(), PH_RESULT_SUCCESS);
    const profiler_hub::sql_cancel_scope scope{ future.get() };
    EXPECT_THROW(loaded.events(7, 0, 0), profiler_hub::operation_cancelled);
    EXPECT_THROW(loaded.events(7, 15, 16), profiler_hub::operation_cancelled);

    std::ofstream{ fixture.trace, std::ios::app } << "x";
    track_file_store stale;
    ASSERT_TRUE(stale.open(fixture.config.string(), sources, 1, 2, 0, 0));
    EXPECT_FALSE(stale.ready());
}
