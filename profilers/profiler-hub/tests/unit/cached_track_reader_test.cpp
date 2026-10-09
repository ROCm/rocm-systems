// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cached_track_reader.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace profiler_hub;
using reader_types::event_type_t;
using reader_types::track_kind_t;

constexpr size_t int64_max =
    static_cast<size_t>(std::numeric_limits<std::int64_t>::max());

struct canned_row
{
    size_t      id;
    uint64_t    start;
    uint64_t    end;
    const char* name;
};

struct fake_state
{
    std::mutex mutex;
    int        events_calls{ 0 };
    int        counter_calls{ 0 };
    int        visit_calls{ 0 };

    std::optional<reader_types::event_filter_t> last_event_filter;
    std::optional<reader_types::event_filter_t> last_counter_filter;

    reader_types::timeline_event_list_t               events;
    reader_types::counter_timeline_event_list_t       counters;
    std::map<event_type_t, std::pair<size_t, size_t>> spans;
    std::map<event_type_t, std::vector<canned_row>>   rows;

    bool                      fail_events{ false };
    bool                      fail_visit{ false };
    std::chrono::milliseconds events_delay{ 0 };
};

class fake_reader final : public track_row_reader
{
public:
    explicit fake_reader(fake_state& state)
    : m_state{ state }
    {}

    reader_types::timeline_event_list_t events_for_track(
        const reader_types::track_info_ptr_t&,
        const reader_types::event_filter_t& filter) override
    {
        std::chrono::milliseconds delay;
        {
            const std::scoped_lock lock{ m_state.mutex };
            ++m_state.events_calls;
            m_state.last_event_filter = filter;
            if(m_state.fail_events) throw std::runtime_error("events read failed");
            delay = m_state.events_delay;
        }
        std::this_thread::sleep_for(delay);
        const std::scoped_lock lock{ m_state.mutex };
        return m_state.events;
    }

    reader_types::counter_timeline_event_list_t counter_events_for_track(
        const reader_types::track_info_ptr_t&,
        const reader_types::event_filter_t& filter) override
    {
        const std::scoped_lock lock{ m_state.mutex };
        ++m_state.counter_calls;
        m_state.last_counter_filter = filter;
        return m_state.counters;
    }

    std::optional<std::pair<size_t, size_t>> event_id_span(event_type_t type) override
    {
        const std::scoped_lock lock{ m_state.mutex };
        const auto             it = m_state.spans.find(type);
        if(it == m_state.spans.end()) return std::nullopt;
        return it->second;
    }

    void visit_events_in_id_range(const reader_types::track_info_ptr_t&,
                                  event_type_t              type,
                                  size_t                    id_begin,
                                  size_t                    id_end,
                                  reader_t::event_visitor_t visitor,
                                  void*                     context) override
    {
        std::vector<canned_row> selected;
        {
            const std::scoped_lock lock{ m_state.mutex };
            ++m_state.visit_calls;
            if(m_state.fail_visit) throw std::runtime_error("range read failed");
            for(const auto& row : m_state.rows[type])
            {
                if(row.id >= id_begin && row.id < id_end) selected.push_back(row);
            }
        }
        for(const auto& row : selected)
        {
            visitor(context, { row.id, type }, row.start, row.end, row.name);
        }
    }

private:
    fake_state& m_state;
};

class fake_source final : public connection_source
{
public:
    explicit fake_source(fake_state& state)
    : m_state{ state }
    {}

    std::unique_ptr<track_row_reader> acquire() override
    {
        return std::make_unique<fake_reader>(m_state);
    }

    std::unique_ptr<track_row_reader> try_acquire() override
    {
        return std::make_unique<fake_reader>(m_state);
    }

private:
    fake_state& m_state;
};

reader_types::timeline_event_t
timeline_event(uint64_t     start,
               uint64_t     end,
               const char*  name,
               size_t       id   = 0,
               event_type_t type = event_type_t::region)
{
    reader_types::timeline_event_t event;
    event.unique_identifier = { id, type };
    event.start_timestamp   = start;
    event.end_timestamp     = end;
    event.display_name      = name;
    return event;
}

reader_types::counter_timeline_event_t
counter_event(uint64_t timestamp, double value)
{
    reader_types::counter_timeline_event_t event;
    event.timestamp = timestamp;
    event.value     = value;
    return event;
}

reader_types::track_info_ptr_t
make_track(size_t id, track_kind_t kind, size_t event_count = 3)
{
    auto track         = std::make_shared<reader_types::track_info_t>();
    track->id          = id;
    track->category    = kind;
    track->event_count = event_count;
    return track;
}

std::vector<uint64_t>
starts_of(const ph_event_list_t& list)
{
    std::vector<uint64_t> starts;
    for(uint32_t i = 0; i < list.list_size; ++i)
    {
        starts.push_back(list.events[i].start);
    }
    return starts;
}

std::vector<std::string>
names_of(const ph_event_list_t& list)
{
    std::vector<std::string> names;
    for(uint32_t i = 0; i < list.list_size; ++i)
    {
        names.emplace_back(list.events[i].name);
    }
    return names;
}

class cached_track_reader_test : public ::testing::Test
{
protected:
    [[nodiscard]] cached_track_reader make_reader(track_read_options options = {})
    {
        return cached_track_reader{ m_source, m_workers, options };
    }

    void give_three_unsorted_events()
    {
        m_state.events = { timeline_event(30, 31, "c"),
                           timeline_event(10, 11, "a"),
                           timeline_event(20, 21, "b") };
    }

    fake_state          m_state;
    fake_source         m_source{ m_state };
    common::thread_pool m_workers{ 3 };
};

TEST_F(cached_track_reader_test, whole_track_events_are_sorted_by_start)
{
    give_three_unsorted_events();
    auto reader = make_reader();

    const auto list = reader.events(make_track(1, track_kind_t::thread), 0, 0);

    EXPECT_THAT(starts_of(list), ::testing::ElementsAre(10, 20, 30));
    EXPECT_THAT(names_of(list), ::testing::ElementsAre("a", "b", "c"));
}

TEST_F(cached_track_reader_test,
       whole_track_events_are_read_once_and_the_storage_is_shared)
{
    give_three_unsorted_events();
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    const auto first  = reader.events(track, 0, 0);
    const auto second = reader.events(track, 0, 0);

    EXPECT_EQ(m_state.events_calls, 1);
    EXPECT_EQ(first.events, second.events);
    EXPECT_EQ(first.list_size, second.list_size);
}

TEST_F(cached_track_reader_test, different_tracks_are_cached_separately)
{
    give_three_unsorted_events();
    auto reader = make_reader();

    const auto first  = reader.events(make_track(1, track_kind_t::thread), 0, 0);
    const auto second = reader.events(make_track(2, track_kind_t::thread), 0, 0);

    EXPECT_EQ(m_state.events_calls, 2);
    EXPECT_NE(first.events, second.events);
}

TEST_F(cached_track_reader_test, concurrent_first_requests_read_the_track_once)
{
    give_three_unsorted_events();
    m_state.events_delay = std::chrono::milliseconds{ 50 };
    auto       reader    = make_reader();
    const auto track     = make_track(1, track_kind_t::thread);

    constexpr int                requesters = 8;
    std::vector<ph_event_list_t> lists(requesters);
    std::vector<std::thread>     threads;
    for(int i = 0; i < requesters; ++i)
    {
        threads.emplace_back([&, i] { lists[i] = reader.events(track, 0, 0); });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(m_state.events_calls, 1);
    for(const auto& list : lists)
    {
        EXPECT_EQ(list.events, lists[0].events);
        EXPECT_EQ(list.list_size, 3U);
    }
}

TEST_F(cached_track_reader_test, a_failed_read_is_not_cached)
{
    give_three_unsorted_events();
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    m_state.fail_events = true;
    EXPECT_THROW(std::ignore = reader.events(track, 0, 0), std::runtime_error);

    m_state.fail_events = false;
    const auto list     = reader.events(track, 0, 0);

    EXPECT_EQ(list.list_size, 3U);
    EXPECT_EQ(m_state.events_calls, 2);
}

TEST_F(cached_track_reader_test, a_windowed_request_is_served_from_the_whole_track_read)
{
    give_three_unsorted_events();
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    const auto windowed = reader.events(track, 15, 25);
    const auto whole    = reader.events(track, 0, 0);

    EXPECT_EQ(m_state.events_calls, 1);
    EXPECT_THAT(starts_of(windowed), ::testing::ElementsAre(20));
    EXPECT_EQ(whole.list_size, 3U);
}

TEST_F(cached_track_reader_test,
       a_window_returns_events_overlapping_it_including_its_edges)
{
    m_state.events    = { timeline_event(0, 100, "long"),
                          timeline_event(10, 20, "ends_at_start"),
                          timeline_event(30, 40, "inside"),
                          timeline_event(50, 60, "starts_at_end"),
                          timeline_event(70, 80, "after") };
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    const auto list = reader.events(track, 20, 50);

    EXPECT_THAT(
        names_of(list),
        ::testing::ElementsAre("long", "ends_at_start", "inside", "starts_at_end"));
}

TEST_F(cached_track_reader_test, a_window_without_an_upper_bound_reaches_the_track_end)
{
    give_three_unsorted_events();
    auto reader = make_reader();

    const auto list = reader.events(make_track(1, track_kind_t::thread), 15, 0);

    EXPECT_THAT(starts_of(list), ::testing::ElementsAre(20, 30));
}

TEST_F(cached_track_reader_test, windowed_results_stay_valid_after_later_requests)
{
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    m_state.events    = { timeline_event(1, 2, "first"),
                          timeline_event(7, 8, "second"),
                          timeline_event(9, 10, "third") };
    const auto first  = reader.events(track, 1, 2);
    const auto second = reader.events(track, 6, 12);

    EXPECT_NE(first.events, second.events);
    ASSERT_EQ(first.list_size, 1U);
    EXPECT_EQ(first.events[0].start, 1U);
    EXPECT_STREQ(first.events[0].name, "first");
    EXPECT_EQ(second.list_size, 2U);
}

TEST_F(cached_track_reader_test, events_carry_their_depth_in_the_whole_track)
{
    m_state.events    = { timeline_event(0, 10, "outer"),
                          timeline_event(2, 5, "inner"),
                          timeline_event(3, 4, "innermost"),
                          timeline_event(6, 8, "later"),
                          timeline_event(10, 12, "touching") };
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    const auto list = reader.events(track, 0, 0);

    ASSERT_EQ(list.list_size, 5U);
    EXPECT_EQ(list.events[0].nesting_depth, 1U);
    EXPECT_EQ(list.events[1].nesting_depth, 2U);
    EXPECT_EQ(list.events[2].nesting_depth, 3U);
    EXPECT_EQ(list.events[3].nesting_depth, 2U);
    EXPECT_EQ(list.events[4].nesting_depth, 1U);
}

TEST_F(cached_track_reader_test, the_nesting_depth_is_the_deepest_event_of_the_track)
{
    m_state.events = { timeline_event(0, 10, "outer"),
                       timeline_event(2, 5, "inner"),
                       timeline_event(3, 4, "innermost"),
                       timeline_event(20, 30, "alone") };
    auto reader    = make_reader();

    EXPECT_EQ(reader.nesting_depth(make_track(1, track_kind_t::thread)), 3U);
}

TEST_F(cached_track_reader_test, the_nesting_depth_shares_the_whole_track_read)
{
    give_three_unsorted_events();
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    std::ignore = reader.events(track, 0, 0);
    std::ignore = reader.nesting_depth(track);
    std::ignore = reader.nesting_depth(track);

    EXPECT_EQ(m_state.events_calls, 1);
}

TEST_F(cached_track_reader_test, a_track_without_events_has_nesting_depth_zero)
{
    auto reader = make_reader();

    EXPECT_EQ(reader.nesting_depth(make_track(1, track_kind_t::thread, 0)), 0U);
}

TEST_F(cached_track_reader_test,
       a_pmc_or_null_track_has_nesting_depth_zero_without_a_read)
{
    give_three_unsorted_events();
    auto reader = make_reader();

    EXPECT_EQ(reader.nesting_depth(make_track(1, track_kind_t::pmc_agent)), 0U);
    EXPECT_EQ(reader.nesting_depth(nullptr), 0U);
    EXPECT_EQ(m_state.events_calls, 0);
}

TEST_F(cached_track_reader_test, a_windowed_event_keeps_its_depth_from_the_whole_track)
{
    m_state.events = { timeline_event(0, 100, "outer"), timeline_event(40, 50, "inner") };
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    const auto list = reader.events(track, 45, 60);

    ASSERT_EQ(list.list_size, 2U);
    EXPECT_EQ(list.events[0].nesting_depth, 1U);
    EXPECT_EQ(list.events[1].nesting_depth, 2U);
}

TEST_F(cached_track_reader_test, events_carry_their_id_and_type)
{
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    m_state.events  = { timeline_event(1, 2, "r", 7, event_type_t::region),
                        timeline_event(3, 4, "k", 8, event_type_t::kernel_dispatch),
                        timeline_event(5, 6, "c", 9, event_type_t::memory_copy),
                        timeline_event(7, 8, "a", 10, event_type_t::memory_allocate) };
    const auto list = reader.events(track, 0, 0);

    ASSERT_EQ(list.list_size, 4U);
    EXPECT_EQ(list.events[0].id, 7U);
    EXPECT_EQ(list.events[0].type, PH_EVENT_TYPE_REGION);
    EXPECT_EQ(list.events[1].id, 8U);
    EXPECT_EQ(list.events[1].type, PH_EVENT_TYPE_KERNEL_DISPATCH);
    EXPECT_EQ(list.events[2].id, 9U);
    EXPECT_EQ(list.events[2].type, PH_EVENT_TYPE_MEMORY_COPY);
    EXPECT_EQ(list.events[3].id, 10U);
    EXPECT_EQ(list.events[3].type, PH_EVENT_TYPE_MEMORY_ALLOCATE);
}

TEST_F(cached_track_reader_test, a_windowed_event_carries_its_id_and_type)
{
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    m_state.events  = { timeline_event(1, 2, "k", 5, event_type_t::kernel_dispatch) };
    const auto list = reader.events(track, 1, 5);

    ASSERT_EQ(list.list_size, 1U);
    EXPECT_EQ(list.events[0].id, 5U);
    EXPECT_EQ(list.events[0].type, PH_EVENT_TYPE_KERNEL_DISPATCH);
}

TEST_F(cached_track_reader_test, an_event_that_is_not_a_duration_event_fails_the_read)
{
    auto       reader = make_reader();
    const auto track  = make_track(1, track_kind_t::thread);

    m_state.events = { timeline_event(1, 2, "s", 1, event_type_t::sample) };

    EXPECT_THROW(std::ignore = reader.events(track, 0, 0), std::invalid_argument);
}

TEST_F(cached_track_reader_test, a_pmc_track_has_no_events_and_the_reader_is_not_used)
{
    give_three_unsorted_events();
    auto reader = make_reader();

    const auto list = reader.events(make_track(1, track_kind_t::pmc_agent), 0, 0);

    EXPECT_EQ(list.list_size, 0U);
    EXPECT_EQ(m_state.events_calls, 0);
}

TEST_F(cached_track_reader_test, a_null_track_has_no_events_and_no_samples)
{
    auto reader = make_reader();

    EXPECT_EQ(reader.events(nullptr, 0, 0).list_size, 0U);
    EXPECT_EQ(reader.samples(nullptr, 0, 0).list_size, 0U);
}

TEST_F(cached_track_reader_test, a_track_that_is_not_pmc_has_no_samples)
{
    m_state.counters = { counter_event(1, 1.0) };
    auto reader      = make_reader();

    const auto list = reader.samples(make_track(1, track_kind_t::thread), 0, 0);

    EXPECT_EQ(list.list_size, 0U);
    EXPECT_EQ(m_state.counter_calls, 0);
}

TEST_F(cached_track_reader_test, whole_track_samples_are_sorted_and_read_once)
{
    m_state.counters  = { counter_event(30, 3.0),
                          counter_event(10, 1.0),
                          counter_event(20, 2.0) };
    auto       reader = make_reader();
    const auto track  = make_track(4, track_kind_t::pmc_agent);

    const auto first  = reader.samples(track, 0, 0);
    const auto second = reader.samples(track, 0, 0);

    ASSERT_EQ(first.list_size, 3U);
    EXPECT_EQ(first.samples[0].timestamp, 10U);
    EXPECT_EQ(first.samples[1].timestamp, 20U);
    EXPECT_EQ(first.samples[2].timestamp, 30U);
    EXPECT_DOUBLE_EQ(first.samples[0].value, 1.0);
    EXPECT_EQ(first.samples, second.samples);
    EXPECT_EQ(m_state.counter_calls, 1);
}

TEST_F(cached_track_reader_test, a_windowed_sample_request_is_read_each_time)
{
    m_state.counters  = { counter_event(10, 1.0) };
    auto       reader = make_reader();
    const auto track  = make_track(4, track_kind_t::pmc_agent);

    std::ignore = reader.samples(track, 5, 15);
    std::ignore = reader.samples(track, 5, 15);

    EXPECT_EQ(m_state.counter_calls, 2);
    ASSERT_TRUE(m_state.last_counter_filter.has_value());
    EXPECT_EQ(m_state.last_counter_filter->time_window.start, 5U);
    EXPECT_EQ(m_state.last_counter_filter->time_window.end, 15U);
}

TEST_F(cached_track_reader_test, a_small_thread_track_is_read_with_one_query)
{
    give_three_unsorted_events();
    track_read_options options;
    options.parallel_read_min_events = 100;
    auto reader                      = make_reader(options);

    const auto list = reader.events(make_track(1, track_kind_t::thread, 3), 0, 0);

    EXPECT_EQ(list.list_size, 3U);
    EXPECT_EQ(m_state.events_calls, 1);
    EXPECT_EQ(m_state.visit_calls, 0);
}

class large_thread_track_test : public cached_track_reader_test
{
protected:
    void SetUp() override
    {
        m_state.spans[event_type_t::region]          = { 1, 6 };
        m_state.spans[event_type_t::kernel_dispatch] = { 1, 4 };
        m_state.rows[event_type_t::region] = { { 1, 60, 61, "r1" }, { 2, 50, 51, "r2" },
                                               { 3, 40, 41, "r3" }, { 4, 30, 31, "r4" },
                                               { 5, 20, 21, "r5" }, { 6, 10, 11, "r6" } };
        m_state.rows[event_type_t::kernel_dispatch] = { { 1, 55, 56, "k1" },
                                                        { 2, 45, 46, "k2" },
                                                        { 3, 35, 36, "k3" },
                                                        { 4, 25, 26, "k4" } };
        m_options.parallel_read_min_events          = 1;
        m_options.parallel_read_parts               = 3;
    }

    track_read_options m_options;
};

TEST_F(large_thread_track_test, is_read_in_parts_and_merged_in_start_order)
{
    auto reader = make_reader(m_options);

    const auto list = reader.events(make_track(1, track_kind_t::thread, 10), 0, 0);

    EXPECT_THAT(starts_of(list),
                ::testing::ElementsAre(10, 20, 25, 30, 35, 40, 45, 50, 55, 60));
    EXPECT_THAT(names_of(list),
                ::testing::ElementsAre(
                    "r6", "r5", "k4", "r4", "k3", "r3", "k2", "r2", "k1", "r1"));
    EXPECT_EQ(m_state.visit_calls, 3);
    EXPECT_EQ(m_state.events_calls, 0);
}

TEST_F(large_thread_track_test, parts_carry_the_id_and_type_of_every_event)
{
    auto reader = make_reader(m_options);

    const auto list = reader.events(make_track(1, track_kind_t::thread, 10), 0, 0);

    ASSERT_EQ(list.list_size, 10U);
    EXPECT_EQ(list.events[0].id, 6U);
    EXPECT_EQ(list.events[0].type, PH_EVENT_TYPE_REGION);
    EXPECT_EQ(list.events[2].id, 4U);
    EXPECT_EQ(list.events[2].type, PH_EVENT_TYPE_KERNEL_DISPATCH);
}

TEST_F(large_thread_track_test, is_read_once_and_the_storage_is_shared)
{
    auto       reader = make_reader(m_options);
    const auto track  = make_track(1, track_kind_t::thread, 10);

    const auto first  = reader.events(track, 0, 0);
    const auto second = reader.events(track, 0, 0);

    EXPECT_EQ(first.events, second.events);
    EXPECT_EQ(m_state.visit_calls, 3);
}

TEST_F(large_thread_track_test, only_thread_tracks_are_read_in_parts)
{
    give_three_unsorted_events();
    auto reader = make_reader(m_options);

    const auto list = reader.events(make_track(2, track_kind_t::stream, 10), 0, 0);

    EXPECT_EQ(list.list_size, 3U);
    EXPECT_EQ(m_state.visit_calls, 0);
    EXPECT_EQ(m_state.events_calls, 1);
}

TEST_F(large_thread_track_test, a_failing_part_fails_the_read_and_the_next_read_works)
{
    auto       reader = make_reader(m_options);
    const auto track  = make_track(1, track_kind_t::thread, 10);

    m_state.fail_visit = true;
    EXPECT_THROW(std::ignore = reader.events(track, 0, 0), std::runtime_error);

    m_state.fail_visit = false;
    const auto list    = reader.events(track, 0, 0);

    EXPECT_EQ(list.list_size, 10U);
}

TEST_F(large_thread_track_test,
       a_single_part_reads_every_table_range_on_the_calling_thread)
{
    m_options.parallel_read_parts = 1;
    auto reader                   = make_reader(m_options);

    const auto list = reader.events(make_track(1, track_kind_t::thread, 10), 0, 0);

    EXPECT_EQ(list.list_size, 10U);
    EXPECT_EQ(m_state.visit_calls, 2);
}

TEST_F(large_thread_track_test, tables_without_rows_are_skipped)
{
    m_state.spans.erase(event_type_t::kernel_dispatch);
    auto reader = make_reader(m_options);

    const auto list = reader.events(make_track(1, track_kind_t::thread, 6), 0, 0);

    EXPECT_THAT(starts_of(list), ::testing::ElementsAre(10, 20, 30, 40, 50, 60));
}

TEST_F(large_thread_track_test, the_nesting_depth_is_read_from_the_merged_track)
{
    auto reader = make_reader(m_options);

    EXPECT_EQ(reader.nesting_depth(make_track(1, track_kind_t::thread, 10)), 1U);
    EXPECT_EQ(m_state.visit_calls, 3);
}

TEST_F(large_thread_track_test, parts_get_their_depth_after_the_merge)
{
    auto reader = make_reader(m_options);

    const auto list = reader.events(make_track(1, track_kind_t::thread, 10), 0, 0);

    ASSERT_EQ(list.list_size, 10U);
    for(uint32_t i = 0; i < list.list_size; ++i)
    {
        EXPECT_EQ(list.events[i].nesting_depth, 1U);
    }
}

}  // namespace
