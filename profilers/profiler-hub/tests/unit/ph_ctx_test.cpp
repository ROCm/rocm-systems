// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ph_ctx.hpp"

#include "profiler-hub/cpp/storage.hpp"
#include "profiler-hub/cpp/writer.hpp"
#include "profiler-hub/cpp/writer_types.hpp"
#include "rocpd_v2_fixture.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{

using namespace profiler_hub;

class ph_ctx_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "ph_ctx_test_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer() const
    {
        return std::make_unique<writer_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    // Registers node/process/thread, a track matching that topology, and one
    // region event on it, so ph_ctx sees exactly one non-empty track.
    void seed_track_with_one_event(writer_t& writer) const
    {
        const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
        writer.register_node_info(node_info);

        writer_types::process_info_t process_info;
        process_info.pid     = 100;
        process_info.node_id = 1;
        writer.register_process_info(process_info);

        writer_types::thread_info_t thread_info;
        thread_info.thread_id  = 200;
        thread_info.node_id    = 1;
        thread_info.process_id = 100;
        writer.register_thread_info(thread_info);

        writer_types::track_info_t track_info;
        track_info.node_id    = 1;
        track_info.process_id = 100;
        track_info.thread_id  = 200;
        writer.register_track_info(track_info);

        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 100;
        trace_environment.thread_id  = 200;

        writer_types::event_data_t event_data;
        event_data.stack_id = 1;

        writer_types::region_data_t region_data;
        region_data.name            = "test-region";
        region_data.start_timestamp = 1000;
        region_data.end_timestamp   = 2000;
        region_data.event           = event_data;
        writer.insert_region_data(region_data, trace_environment);
    }

    void seed_track_with_two_events(writer_t& writer) const
    {
        const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
        writer.register_node_info(node_info);

        writer_types::process_info_t process_info;
        process_info.pid     = 100;
        process_info.node_id = 1;
        writer.register_process_info(process_info);

        writer_types::thread_info_t thread_info;
        thread_info.thread_id  = 200;
        thread_info.node_id    = 1;
        thread_info.process_id = 100;
        writer.register_thread_info(thread_info);

        writer_types::track_info_t track_info;
        track_info.node_id    = 1;
        track_info.process_id = 100;
        track_info.thread_id  = 200;
        writer.register_track_info(track_info);

        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 100;
        trace_environment.thread_id  = 200;

        writer_types::event_data_t event_data;
        event_data.stack_id = 1;

        writer_types::region_data_t region_a;
        region_a.name            = "region-a";
        region_a.start_timestamp = 1000;
        region_a.end_timestamp   = 2000;
        region_a.event           = event_data;
        writer.insert_region_data(region_a, trace_environment);

        writer_types::region_data_t region_b;
        region_b.name            = "region-b";
        region_b.start_timestamp = 5000;
        region_b.end_timestamp   = 6000;
        region_b.event           = event_data;
        writer.insert_region_data(region_b, trace_environment);
    }

    std::string m_db_path;
    // Embedded verbatim into unquoted SQL table names by insert_statements, so it
    // must be a valid identifier fragment - no hyphens.
    std::string m_uuid = "testuuid0000";
};

TEST_F(ph_ctx_test, construction_populates_schema_version_and_node)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto version = ctx.get_schema_version();
    EXPECT_GT(version.major + version.minor + version.patch, 0U);

    const auto node = ctx.get_node();
    EXPECT_EQ(node.info.id, 1U);
    EXPECT_STREQ(node.info.machine_id, "machine-1");
}

TEST_F(ph_ctx_test, construction_with_no_nodes_succeeds_with_empty_node_info)
{
    auto writer = make_writer();
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto node = ctx.get_node();
    EXPECT_EQ(node.info.id, 0U);
    ASSERT_NE(node.info.machine_id, nullptr);
    EXPECT_STREQ(node.info.machine_id, "");
    ASSERT_NE(node.info.system_name, nullptr);
    EXPECT_STREQ(node.info.system_name, "");
    ASSERT_NE(node.info.hostname, nullptr);
    EXPECT_STREQ(node.info.hostname, "");
    ASSERT_NE(node.info.release, nullptr);
    EXPECT_STREQ(node.info.release, "");
    ASSERT_NE(node.info.version, nullptr);
    EXPECT_STREQ(node.info.version, "");
    ASSERT_NE(node.info.hardware_name, nullptr);
    EXPECT_STREQ(node.info.hardware_name, "");
    ASSERT_NE(node.info.domain_name, nullptr);
    EXPECT_STREQ(node.info.domain_name, "");
}

TEST_F(ph_ctx_test, get_track_list_returns_seeded_non_empty_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);
    EXPECT_EQ(tracks.tracks[0].nid, 1U);
    EXPECT_EQ(tracks.tracks[0].pid, 100U);
    EXPECT_EQ(tracks.tracks[0].tid, 200U);
    EXPECT_EQ(tracks.tracks[0].event_count, 1U);
}

TEST_F(ph_ctx_test, has_track_reflects_seeded_and_unknown_ids)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    EXPECT_TRUE(ctx.has_track(tracks.tracks[0].id));
    EXPECT_FALSE(ctx.has_track(tracks.tracks[0].id + 1000));
}

TEST_F(ph_ctx_test, get_track_events_returns_event_for_seeded_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto events = ctx.get_track_events(tracks.tracks[0].id, 0, 0);
    ASSERT_EQ(events.list_size, 1U);
    EXPECT_EQ(events.events[0].start, 1000U);
    EXPECT_EQ(events.events[0].end, 2000U);
}

TEST_F(ph_ctx_test, get_track_events_returns_empty_for_unknown_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto events = ctx.get_track_events(999999, 0, 0);
    EXPECT_EQ(events.list_size, 0U);
    EXPECT_EQ(events.events, nullptr);
}

TEST_F(ph_ctx_test, get_track_events_start_only_filters_to_overlapping_events)
{
    auto writer = make_writer();
    seed_track_with_two_events(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto events = ctx.get_track_events(tracks.tracks[0].id, 3000, 0);
    ASSERT_EQ(events.list_size, 1U);
    EXPECT_EQ(events.events[0].start, 5000U);
    EXPECT_EQ(events.events[0].end, 6000U);
}

TEST_F(ph_ctx_test, get_track_events_end_only_filters_to_overlapping_events)
{
    auto writer = make_writer();
    seed_track_with_two_events(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto events = ctx.get_track_events(tracks.tracks[0].id, 0, 3000);
    ASSERT_EQ(events.list_size, 1U);
    EXPECT_EQ(events.events[0].start, 1000U);
    EXPECT_EQ(events.events[0].end, 2000U);
}

TEST_F(ph_ctx_test, get_track_events_both_bounds_filters_to_overlapping_events)
{
    auto writer = make_writer();
    seed_track_with_two_events(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto events = ctx.get_track_events(tracks.tracks[0].id, 1200, 1800);
    ASSERT_EQ(events.list_size, 1U);
    EXPECT_EQ(events.events[0].start, 1000U);
    EXPECT_EQ(events.events[0].end, 2000U);
}

TEST_F(ph_ctx_test, get_track_events_no_bounds_returns_all_events)
{
    auto writer = make_writer();
    seed_track_with_two_events(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto events = ctx.get_track_events(tracks.tracks[0].id, 0, 0);
    EXPECT_EQ(events.list_size, 2U);
}

TEST_F(ph_ctx_test, get_track_samples_returns_empty_for_non_pmc_track)
{
    auto writer = make_writer();
    seed_track_with_one_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ph_ctx ctx{ m_db_path };
    ctx.load();

    const auto tracks = ctx.get_track_list();
    ASSERT_EQ(tracks.list_size, 1U);

    const auto samples = ctx.get_track_samples(tracks.tracks[0].id, 0, 0);
    EXPECT_EQ(samples.list_size, 0U);
}

const ph_track_t*
find_track(const ph_track_list_t& tracks, ph_track_category_t category)
{
    for(uint32_t index = 0; index < tracks.list_size; ++index)
    {
        if(tracks.tracks[index].category == category) return &tracks.tracks[index];
    }
    return nullptr;
}

const ph_event_t*
find_event(const ph_event_list_t& events, ph_event_type_t type, uint64_t id)
{
    for(uint32_t index = 0; index < events.list_size; ++index)
    {
        if(events.events[index].type == type && events.events[index].id == id)
        {
            return &events.events[index];
        }
    }
    return nullptr;
}

// The schema 3 writer fixture above has no agents, queues, kernels, copies, or
// counters. The schema 2 trace has all of them, so it reaches the node
// converters and the detail, histogram, query, and trim paths.
TEST(ph_ctx_schema2_test, load_exposes_node_lists_details_histogram_queries_and_trim)
{
    const auto path = std::filesystem::temp_directory_path() / "ph_ctx_schema2_test.rpd";
    const auto trimmed =
        std::filesystem::temp_directory_path() / "ph_ctx_schema2_test.trimmed.rpd";
    std::filesystem::remove(path);
    std::filesystem::remove(trimmed);
    test::write_schema2(path);

    {
        ph_ctx ctx{ path.string() };
        ctx.load();
        EXPECT_EQ(ctx.wait_until_ready(), PH_RESULT_SUCCESS);
        EXPECT_EQ(ctx.trace_source_count(), 1U);
        EXPECT_NE(ctx.trace_source_path(0), nullptr);
        EXPECT_EQ(ctx.get_schema_version().major, 2U);
        EXPECT_NE(ctx.catalog(), nullptr);

        const ph_node_t node = ctx.source_node(0);
        EXPECT_GE(node.agents.list_size, 1U);
        EXPECT_GE(node.process_list.list_size, 1U);
        EXPECT_GE(node.thread_list.list_size, 1U);
        EXPECT_GE(node.queue_list.list_size, 1U);
        EXPECT_GE(node.pmc_info_list.list_size, 1U);

        const auto  tracks = ctx.get_track_list();
        const auto* thread = find_track(tracks, PH_TRACK_CATEGORY_THREAD);
        const auto* kernel =
            find_track(tracks, PH_TRACK_CATEGORY_KERNEL_DISPATCH_AGENT_QUEUE);
        const auto* copy = find_track(tracks, PH_TRACK_CATEGORY_MEMORY_COPY_AGENT_QUEUE);
        const auto* pmc  = find_track(tracks, PH_TRACK_CATEGORY_PMC_AGENT);
        ASSERT_NE(thread, nullptr);
        ASSERT_NE(kernel, nullptr);
        ASSERT_NE(copy, nullptr);
        ASSERT_NE(pmc, nullptr);
        EXPECT_TRUE(ctx.has_track(thread->id));

        auto        regions = ctx.get_track_events(thread->id, 0, 0);
        const auto* region  = find_event(regions, PH_EVENT_TYPE_REGION, 2);
        ASSERT_NE(region, nullptr);

        const auto region_details = ctx.get_region_details(thread->id, region->id);
        EXPECT_EQ(region_details.is_valid, 1U);
        EXPECT_EQ(ctx.get_region_details(999999, region->id).is_valid, 0U);

        const auto stack =
            ctx.get_call_stack(thread->id, region->id, PH_EVENT_TYPE_REGION);
        EXPECT_GE(stack.list_size, 1U);
        EXPECT_EQ(
            ctx.get_call_stack(thread->id, region->id, static_cast<ph_event_type_t>(255))
                .list_size,
            0U);

        const auto correlated =
            ctx.get_correlated_events(thread->id, region->id, PH_EVENT_TYPE_REGION);
        ASSERT_TRUE(correlated.has_value());
        EXPECT_GE(correlated->list_size, 1U);
        const auto resolved = ctx.track_id_of(999999, region->id, PH_EVENT_TYPE_REGION);
        ASSERT_TRUE(resolved.has_value());
        EXPECT_EQ(resolved.value(), thread->id);

        auto        kernels      = ctx.get_track_events(kernel->id, 0, 0);
        const auto* kernel_event = find_event(kernels, PH_EVENT_TYPE_KERNEL_DISPATCH, 10);
        ASSERT_NE(kernel_event, nullptr);
        const auto dispatch =
            ctx.get_kernel_dispatch_details(kernel->id, kernel_event->id);
        EXPECT_EQ(dispatch.is_valid, 1U);
        EXPECT_EQ(dispatch.grid_size_x, 8U);
        EXPECT_EQ(ctx.get_memory_alloc_details(kernel->id, kernel_event->id).is_valid,
                  0U);

        auto        copies     = ctx.get_track_events(copy->id, 0, 0);
        const auto* copy_event = find_event(copies, PH_EVENT_TYPE_MEMORY_COPY, 11);
        ASSERT_NE(copy_event, nullptr);
        const auto memory = ctx.get_memory_copy_details(copy->id, copy_event->id);
        EXPECT_EQ(memory.is_valid, 1U);
        EXPECT_EQ(memory.size, 4096U);

        auto samples = ctx.get_track_samples(pmc->id, 0, 0);
        EXPECT_EQ(samples.list_size, 2U);
        EXPECT_EQ(ctx.get_track_samples(999999, 0, 0).list_size, 0U);

        ph_histogram_t histogram{};
        ctx.build_histogram(8, histogram);
        EXPECT_GT(histogram.track_count, 0U);
        EXPECT_GT(histogram.bucket_count, 0U);

        profiler_hub::resolved_event_query query;
        query.source_index = 0;
        ph_table_event rows[8]{};
        EXPECT_GE(ctx.read_table_events(
                      query, PH_EVENT_TYPE_KERNEL_DISPATCH, 0, UINT64_MAX, rows, 8),
                  1U);
        query.search_strings    = { "vector_add" };
        query.has_string_filter = true;
        query.match_substring   = true;
        EXPECT_GE(ctx.read_table_events(
                      query, PH_EVENT_TYPE_KERNEL_DISPATCH, 0, UINT64_MAX, rows, 8),
                  1U);

        profiler_hub::resolved_event_query missing_track;
        missing_track.tracks = { 999999 };
        EXPECT_EQ(ctx.read_table_events(
                      missing_track, PH_EVENT_TYPE_REGION, 0, UINT64_MAX, rows, 8),
                  0U);

        profiler_hub::resolved_event_query summary_query;
        summary_query.source_index = 0;
        std::vector<profiler_hub::name_duration_summary_t> summaries;
        EXPECT_TRUE(
            ctx.summarize_name_durations(summary_query, PH_EVENT_TYPE_REGION, summaries));
        EXPECT_FALSE(summaries.empty());

        uint64_t first = 0;
        uint64_t last  = 0;
        EXPECT_TRUE(ctx.table_event_id_span(query, PH_EVENT_TYPE_REGION, first, last));
        EXPECT_LE(first, last);

        ctx.release_track_events(&regions);
        ctx.release_track_events(nullptr);
        ctx.release_track_samples(&samples);
        ctx.release_track_samples(nullptr);
        EXPECT_EQ(regions.list_size, 0U);

        EXPECT_EQ(ctx.trim_trace(160, 170, nullptr, nullptr), PH_RESULT_INVALID_ARGUMENT);
        EXPECT_EQ(ctx.trim_trace(160, 170, "", nullptr), PH_RESULT_INVALID_ARGUMENT);
        ASSERT_EQ(ctx.trim_trace(160, 170, trimmed.string().c_str(), nullptr),
                  PH_RESULT_SUCCESS);

        ph_ctx trimmed_ctx{ trimmed.string() };
        trimmed_ctx.load();
        const auto  trimmed_tracks = trimmed_ctx.get_track_list();
        const auto* trimmed_kernel =
            find_track(trimmed_tracks, PH_TRACK_CATEGORY_KERNEL_DISPATCH_AGENT_QUEUE);
        const auto* trimmed_copy =
            find_track(trimmed_tracks, PH_TRACK_CATEGORY_MEMORY_COPY_AGENT_QUEUE);
        ASSERT_NE(trimmed_kernel, nullptr);
        EXPECT_EQ(trimmed_ctx.get_track_events(trimmed_kernel->id, 0, 0).list_size, 1U);
        if(trimmed_copy != nullptr)
        {
            EXPECT_EQ(trimmed_ctx.get_track_events(trimmed_copy->id, 0, 0).list_size, 0U);
        }
    }

    std::filesystem::remove(path);
    std::filesystem::remove(trimmed);
}

}  // namespace
