// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/cpp/reader.hpp"
#include "trace_fixtures.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace profiler_hub;
using reader_types::event_type_t;

struct visited_event
{
    uint64_t    start;
    uint64_t    end;
    std::string name;
};

class reader_id_range_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = test::temp_trace_path("reader_id_range_test");
        test::write_thread_track_with_two_regions(m_db_path);
        m_reader = std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, ""));

        const auto tracks = m_reader->get_all_tracks();
        ASSERT_EQ(tracks.size(), 1U);
        m_track = tracks.front();
    }

    void TearDown() override
    {
        m_reader.reset();
        std::filesystem::remove(m_db_path);
    }

    std::vector<visited_event> visit(event_type_t                          type,
                                     size_t                                begin,
                                     size_t                                end,
                                     const reader_types::track_info_ptr_t& track) const
    {
        std::vector<visited_event> visited;
        m_reader->visit_track_events_in_id_range(
            track,
            type,
            begin,
            end,
            [](void*                        context,
               reader_types::timestamp_ns_t start,
               reader_types::timestamp_ns_t end,
               std::string_view             name) {
                static_cast<std::vector<visited_event>*>(context)->push_back(
                    { start, end, std::string{ name } });
            },
            &visited);
        return visited;
    }

    static std::vector<std::string> names_of(const std::vector<visited_event>& events)
    {
        std::vector<std::string> names;
        for(const auto& event : events)
        {
            names.push_back(event.name);
        }
        return names;
    }

    std::string                    m_db_path;
    std::unique_ptr<reader_t>      m_reader;
    reader_types::track_info_ptr_t m_track;
};

TEST_F(reader_id_range_test, the_region_table_has_an_id_span_and_the_others_do_not)
{
    const auto region = m_reader->get_event_id_span(event_type_t::region);

    ASSERT_TRUE(region.has_value());
    EXPECT_LT(region->first, region->second);
    EXPECT_FALSE(m_reader->get_event_id_span(event_type_t::kernel_dispatch).has_value());
    EXPECT_FALSE(m_reader->get_event_id_span(event_type_t::memory_copy).has_value());
    EXPECT_FALSE(m_reader->get_event_id_span(event_type_t::memory_allocate).has_value());
}

TEST_F(reader_id_range_test, a_range_over_the_whole_span_visits_every_event_in_id_order)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());

    const auto visited =
        visit(event_type_t::region, span->first, span->second + 1, m_track);

    EXPECT_THAT(names_of(visited), ::testing::ElementsAre("region-a", "region-b"));
    EXPECT_EQ(visited[0].start, 1000U);
    EXPECT_EQ(visited[0].end, 2000U);
}

TEST_F(reader_id_range_test, the_end_of_a_range_is_exclusive)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());

    const auto first_only =
        visit(event_type_t::region, span->first, span->first + 1, m_track);
    const auto second_only =
        visit(event_type_t::region, span->second, span->second + 1, m_track);

    EXPECT_THAT(names_of(first_only), ::testing::ElementsAre("region-a"));
    EXPECT_THAT(names_of(second_only), ::testing::ElementsAre("region-b"));
}

TEST_F(reader_id_range_test, adjacent_ranges_visit_every_event_exactly_once)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());

    auto lower = visit(event_type_t::region, span->first, span->first + 1, m_track);
    auto upper = visit(event_type_t::region, span->first + 1, span->second + 1, m_track);
    lower.insert(lower.end(), upper.begin(), upper.end());

    EXPECT_THAT(names_of(lower), ::testing::ElementsAre("region-a", "region-b"));
}

TEST_F(reader_id_range_test, an_empty_range_visits_nothing)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());

    EXPECT_TRUE(visit(event_type_t::region, span->first, span->first, m_track).empty());
}

TEST_F(reader_id_range_test, a_range_outside_the_span_visits_nothing)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());

    EXPECT_TRUE(visit(event_type_t::region, span->second + 1, span->second + 100, m_track)
                    .empty());
}

TEST_F(reader_id_range_test, another_event_table_visits_nothing)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());

    EXPECT_TRUE(
        visit(event_type_t::kernel_dispatch, span->first, span->second + 1, m_track)
            .empty());
}

TEST_F(reader_id_range_test, a_track_the_catalog_does_not_know_visits_nothing)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());
    const auto unknown = std::make_shared<reader_types::track_info_t>();

    EXPECT_TRUE(
        visit(event_type_t::region, span->first, span->second + 1, unknown).empty());
}

TEST_F(reader_id_range_test, visiting_every_id_matches_reading_the_track_in_one_query)
{
    const auto span = m_reader->get_event_id_span(event_type_t::region);
    ASSERT_TRUE(span.has_value());

    const auto visited =
        visit(event_type_t::region, span->first, span->second + 1, m_track);
    const auto events = m_reader->get_events_for_track(m_track);

    std::vector<std::pair<uint64_t, uint64_t>> by_visit;
    std::vector<std::pair<uint64_t, uint64_t>> by_query;
    for(const auto& event : visited)
    {
        by_visit.emplace_back(event.start, event.end);
    }
    for(const auto& event : events)
    {
        by_query.emplace_back(event.start_timestamp, event.end_timestamp);
    }
    EXPECT_THAT(by_visit, ::testing::UnorderedElementsAreArray(by_query));
}

}  // namespace
