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
#include <tuple>
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

class reader_id_range_all_tables_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = test::temp_trace_path("reader_id_range_all_tables_test");
        std::filesystem::remove(m_db_path);
        {
            auto writer = test::make_trace_writer(m_db_path);
            seed_all_event_tables(*writer);
            writer->flush_in_memory_data_to_disk();
        }
        m_reader = std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, ""));

        for(const auto& track : m_reader->get_all_tracks())
        {
            if(track->category == reader_types::track_kind_t::thread) m_track = track;
        }
        ASSERT_NE(m_track, nullptr);
    }

    void TearDown() override
    {
        m_reader.reset();
        std::filesystem::remove(m_db_path);
    }

    static void seed_all_event_tables(writer_t& writer)
    {
        test::register_thread_track(writer);

        writer_types::agent_info_t agent_info;
        agent_info.unique_id.agent_type = "GPU";
        agent_info.unique_id.type_index = 0;
        agent_info.node_id              = 1;
        agent_info.process_id           = 100;
        writer.register_agent_info(agent_info);

        writer_types::queue_info_t queue_info;
        queue_info.queue_id   = 30;
        queue_info.node_id    = 1;
        queue_info.process_id = 100;
        writer.register_queue_info(queue_info);

        writer_types::stream_info_t stream_info;
        stream_info.stream_id  = 40;
        stream_info.node_id    = 1;
        stream_info.process_id = 100;
        writer.register_stream_info(stream_info);

        writer_types::code_object_info_t code_object_info;
        code_object_info.id         = 10;
        code_object_info.node_id    = 1;
        code_object_info.process_id = 100;
        writer.register_code_object_info(code_object_info);

        writer_types::kernel_symbol_info_t kernel_symbol_info;
        kernel_symbol_info.id          = 20;
        kernel_symbol_info.node_id     = 1;
        kernel_symbol_info.process_id  = 100;
        kernel_symbol_info.code_obj_id = code_object_info.id;
        writer.register_kernel_symbol_info(kernel_symbol_info);

        writer_types::trace_environment_t environment;
        environment.node_id    = 1;
        environment.process_id = 100;
        environment.thread_id  = 200;
        environment.agent_id   = agent_info.unique_id;
        environment.queue_id   = queue_info.queue_id;
        environment.stream_id  = stream_info.stream_id;

        test::insert_region(writer, "region-a", 1000, 2000);
        test::insert_region(writer, "region-b", 5000, 6000);

        writer_types::kernel_dispatch_data_t kernel;
        kernel.kernel_symbol_id = kernel_symbol_info.id;
        kernel.start_timestamp  = 3000;
        kernel.end_timestamp    = 3500;
        kernel.name             = "kernel-k";
        kernel.event            = writer_types::event_data_t{};
        writer.insert_kernel_dispatch_data(kernel, environment);

        writer_types::memory_copy_data_t copy;
        copy.name            = "copy-c";
        copy.region_name     = "copy-c";
        copy.start_timestamp = 7000;
        copy.end_timestamp   = 7500;
        copy.size            = 64;
        copy.event           = writer_types::event_data_t{};
        writer.insert_memory_copy_data(copy, environment);

        writer_types::event_data_t alloc_event;
        alloc_event.event_category = "alloc-cat";
        writer_types::memory_alloc_data_t alloc;
        alloc.type            = "ALLOC";
        alloc.level           = "REAL";
        alloc.start_timestamp = 8000;
        alloc.end_timestamp   = 8200;
        alloc.size            = 64;
        alloc.event           = alloc_event;
        writer.insert_memory_alloc_data(alloc, environment);
    }

    std::vector<visited_event> visit_whole_table(event_type_t type) const
    {
        const auto                 span = m_reader->get_event_id_span(type);
        std::vector<visited_event> visited;
        if(!span.has_value()) return visited;
        m_reader->visit_track_events_in_id_range(
            m_track,
            type,
            span->first,
            span->second + 1,
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

    std::string                    m_db_path;
    std::unique_ptr<reader_t>      m_reader;
    reader_types::track_info_ptr_t m_track;
};

TEST_F(reader_id_range_all_tables_test,
       every_event_table_has_its_events_visited_with_their_names)
{
    const auto region = visit_whole_table(event_type_t::region);
    const auto kernel = visit_whole_table(event_type_t::kernel_dispatch);
    const auto copy   = visit_whole_table(event_type_t::memory_copy);
    const auto alloc  = visit_whole_table(event_type_t::memory_allocate);

    ASSERT_EQ(region.size(), 2U);
    EXPECT_EQ(region[0].name, "region-a");
    ASSERT_EQ(kernel.size(), 1U);
    EXPECT_EQ(kernel[0].name, "kernel-k");
    EXPECT_EQ(kernel[0].start, 3000U);
    EXPECT_EQ(kernel[0].end, 3500U);
    ASSERT_EQ(copy.size(), 1U);
    EXPECT_EQ(copy[0].name, "copy-c");
    EXPECT_EQ(copy[0].start, 7000U);
    ASSERT_EQ(alloc.size(), 1U);
    EXPECT_EQ(alloc[0].name, "alloc-cat");
    EXPECT_EQ(alloc[0].end, 8200U);
}

TEST_F(reader_id_range_all_tables_test,
       visiting_every_table_matches_reading_the_track_in_one_query)
{
    std::vector<std::tuple<uint64_t, uint64_t, std::string>> by_visit;
    for(const auto type : { event_type_t::region,
                            event_type_t::kernel_dispatch,
                            event_type_t::memory_copy,
                            event_type_t::memory_allocate })
    {
        for(const auto& event : visit_whole_table(type))
        {
            by_visit.emplace_back(event.start, event.end, event.name);
        }
    }

    std::vector<std::tuple<uint64_t, uint64_t, std::string>> by_query;
    for(const auto& event : m_reader->get_events_for_track(m_track))
    {
        by_query.emplace_back(event.start_timestamp,
                              event.end_timestamp,
                              std::string{ event.display_name });
    }

    EXPECT_EQ(by_visit.size(), 5U);
    EXPECT_THAT(by_visit, ::testing::UnorderedElementsAreArray(by_query));
}

}  // namespace
