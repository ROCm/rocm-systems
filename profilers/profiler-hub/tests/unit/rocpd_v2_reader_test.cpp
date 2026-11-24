// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_query.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "rocpd_v2_fixture.hpp"
#include "trace_trim.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{

using namespace profiler_hub;
using reader_types::event_type_t;
using reader_types::track_kind_t;

reader_types::track_info_ptr_t
find_track(const reader_types::track_info_list_t& tracks, track_kind_t kind)
{
    for(const auto& track : tracks)
    {
        if(track && track->category == kind) return track;
    }
    return nullptr;
}

const reader_types::timeline_event_t*
find_event(const reader_types::timeline_event_list_t& events,
           event_type_t                               type,
           size_t                                     id)
{
    for(const auto& event : events)
    {
        if(event.unique_identifier.type == type && event.unique_identifier.id == id)
        {
            return &event;
        }
    }
    return nullptr;
}

class rocpd_v2_reader_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_path           = std::filesystem::temp_directory_path() /
                 (std::string{ "rocpd_v2_reader_test_" } + info->name() + ".rpd");
        m_trimmed = m_path;
        m_trimmed.replace_extension(".trimmed.rpd");
        std::filesystem::remove(m_path);
        std::filesystem::remove(m_trimmed);
        test::write_schema2(m_path);
        m_reader =
            std::make_unique<reader_t>(std::make_unique<storage_t>(m_path.string(), ""));
    }

    void TearDown() override
    {
        m_reader.reset();
        std::filesystem::remove(m_path);
        std::filesystem::remove(m_trimmed);
    }

    std::filesystem::path     m_path;
    std::filesystem::path     m_trimmed;
    std::unique_ptr<reader_t> m_reader;
};

}  // namespace

TEST_F(rocpd_v2_reader_test, probe_reports_schema_2)
{
    const auto probe = probe_file(m_path.string());
    EXPECT_EQ(probe.status, PH_RESULT_SUCCESS);
    EXPECT_EQ(probe.trace_schema, reader_types::trace_schema_t::rocpd);
    EXPECT_EQ(probe.schema_version.major, 2U);
}

TEST_F(rocpd_v2_reader_test, catalog_synthesizes_topology_from_api_and_op_keys)
{
    const auto tracks = m_reader->get_all_tracks();
    EXPECT_NE(find_track(tracks, track_kind_t::thread), nullptr);
    EXPECT_NE(find_track(tracks, track_kind_t::kernel_dispatch_agent_queue), nullptr);
    EXPECT_NE(find_track(tracks, track_kind_t::memory_copy_agent_queue), nullptr);
    const auto counter = find_track(tracks, track_kind_t::pmc_agent);
    ASSERT_NE(counter, nullptr);
    EXPECT_EQ(counter->name, "SCLK");
    EXPECT_EQ(counter->agent_id, 1U);
    ASSERT_TRUE(counter->value_range.has_value());
    EXPECT_DOUBLE_EQ(counter->value_range->min, 1.5);
    EXPECT_DOUBLE_EQ(counter->value_range->max, 2.5);
}

TEST_F(rocpd_v2_reader_test, timeline_reads_regions_kernels_copies_and_samples)
{
    const auto  events = m_reader->get_events();
    const auto* kernel = find_event(events, event_type_t::kernel_dispatch, 10);
    const auto* copy   = find_event(events, event_type_t::memory_copy, 11);
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(kernel->display_name, "vector_add");
    EXPECT_EQ(copy->display_name, "CopyHostToDevice");

    reader_types::event_filter_t regions_only;
    regions_only.types = { event_type_t::region };
    EXPECT_EQ(m_reader->get_event_count(regions_only), 3U);

    const auto thread = find_track(m_reader->get_all_tracks(), track_kind_t::thread);
    ASSERT_NE(thread, nullptr);
    int visited = 0;
    m_reader->visit_track_events_in_id_range(
        thread,
        event_type_t::region,
        1,
        3,
        [](void* context,
           reader_types::unique_timeline_event_id_t,
           size_t,
           size_t,
           std::string_view) { ++*static_cast<int*>(context); },
        &visited);
    EXPECT_GT(visited, 0);

    const auto span = m_reader->get_event_id_span(event_type_t::kernel_dispatch);
    ASSERT_TRUE(span.has_value());
    EXPECT_EQ(span->first, 10U);
    EXPECT_EQ(span->second, 10U);

    const auto counter = find_track(m_reader->get_all_tracks(), track_kind_t::pmc_agent);
    const auto samples = m_reader->get_counter_events_for_track(counter, {});
    ASSERT_EQ(samples.size(), 2U);
    EXPECT_DOUBLE_EQ(samples[1].value, 2.5);

    const auto histogram = m_reader->get_histogram(4);
    EXPECT_FALSE(histogram.tracks.empty());

    const auto range = m_reader->get_data_time_range();
    ASSERT_TRUE(range.start.has_value());
    ASSERT_TRUE(range.end.has_value());
    EXPECT_EQ(*range.start, 100U);
    EXPECT_EQ(*range.end, 500U);

    const auto counts = m_reader->get_event_counts();
    EXPECT_EQ(counts.at(event_type_t::region), 3U);
    EXPECT_EQ(counts.at(event_type_t::kernel_dispatch), 1U);
    EXPECT_EQ(counts.at(event_type_t::memory_copy), 1U);
}

TEST_F(rocpd_v2_reader_test, details_call_stack_and_correlation_follow_api_ops)
{
    const auto  events = m_reader->get_events();
    const auto* region = find_event(events, event_type_t::region, 2);
    const auto* kernel = find_event(events, event_type_t::kernel_dispatch, 10);
    const auto* copy   = find_event(events, event_type_t::memory_copy, 11);
    ASSERT_NE(region, nullptr);
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE(copy, nullptr);

    const auto outer =
        m_reader->get_region_details(*find_event(events, event_type_t::region, 1));
    ASSERT_TRUE(outer.has_value());
    EXPECT_EQ(outer->name, "hipLaunchKernel");
    ASSERT_EQ(outer->args.size(), 1U);
    EXPECT_EQ(outer->args[0].value, "grid=1");
    EXPECT_EQ(
        m_reader->get_arguments(*find_event(events, event_type_t::region, 1)).size(), 1U);

    const auto dispatch = m_reader->get_kernel_dispatch_details(*kernel);
    ASSERT_TRUE(dispatch.has_value());
    EXPECT_EQ(dispatch->name, "vector_add");
    EXPECT_EQ(dispatch->grid_size_x, 8U);
    EXPECT_EQ(dispatch->workgroup_size_x, 64U);
    ASSERT_TRUE(dispatch->group_segment_size.has_value());
    EXPECT_EQ(*dispatch->group_segment_size, 256U);

    const auto memory = m_reader->get_memory_copy_details(*copy);
    ASSERT_TRUE(memory.has_value());
    EXPECT_EQ(memory->size, 4096U);
    ASSERT_TRUE(memory->dst_address.has_value());
    EXPECT_EQ(*memory->dst_address, 0x1000U);
    EXPECT_FALSE(memory->src_address.has_value());
    EXPECT_EQ(memory->region_name, "memcpy");

    const auto stack = m_reader->get_call_stack(*region);
    ASSERT_GE(stack.size(), 2U);
    ASSERT_TRUE(stack[0].program_counter.has_value());
    ASSERT_TRUE(stack[1].program_counter.has_value());
    EXPECT_EQ(stack[0].program_counter->function, "memcpy");
    EXPECT_EQ(stack[1].program_counter->function, "hipLaunchKernel");

    const auto from_region = m_reader->get_correlated_events(*region);
    ASSERT_EQ(from_region.size(), 2U);
    EXPECT_EQ(from_region[0].unique_identifier.id, 10U);
    EXPECT_EQ(from_region[1].unique_identifier.id, 11U);
    const auto from_kernel = m_reader->get_correlated_events(*kernel);
    ASSERT_EQ(from_kernel.size(), 1U);
    EXPECT_EQ(from_kernel[0].unique_identifier.id, 2U);
}

TEST_F(rocpd_v2_reader_test, table_events_page_kernels_copies_regions_and_samples)
{
    ph_table_event       rows[8]{};
    resolved_event_query query;
    const uint32_t       kernels = m_reader->read_table_events(
        query, PH_EVENT_TYPE_KERNEL_DISPATCH, 0, UINT64_MAX, rows, 8);
    ASSERT_EQ(kernels, 1U);
    EXPECT_EQ(rows[0].id, 10U);
    EXPECT_EQ(rows[0].grid_size_x, 8U);

    const uint32_t copies = m_reader->read_table_events(
        query, PH_EVENT_TYPE_MEMORY_COPY, 0, UINT64_MAX, rows, 8);
    ASSERT_EQ(copies, 1U);
    EXPECT_EQ(rows[0].id, 11U);
    EXPECT_EQ(rows[0].size, 4096U);

    const uint32_t regions =
        m_reader->read_table_events(query, PH_EVENT_TYPE_REGION, 0, UINT64_MAX, rows, 8);
    EXPECT_EQ(regions, 3U);

    const uint32_t samples = m_reader->read_table_events(
        query, PH_EVENT_TYPE_PMC_SAMPLE, 0, UINT64_MAX, rows, 8);
    EXPECT_EQ(samples, 2U);

    query.node_id = 99;
    EXPECT_EQ(
        m_reader->read_table_events(query, PH_EVENT_TYPE_REGION, 0, UINT64_MAX, rows, 8),
        0U);

    uint64_t first = 0;
    uint64_t last  = 0;
    EXPECT_TRUE(m_reader->table_event_id_span(PH_EVENT_TYPE_REGION, first, last));
    EXPECT_EQ(first, 1U);
    EXPECT_EQ(last, 3U);

    std::vector<name_duration_summary_t> summary;
    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_REGION, summary));
    uint64_t summarized = 0;
    for(const auto& row : summary)
        summarized += row.count;
    EXPECT_EQ(summarized, 3U);
}

TEST_F(rocpd_v2_reader_test, trim_drops_rows_outside_the_window)
{
    m_reader.reset();
    // Window [160, 170): the kernel overlaps it, the copy and the second thread do not.
    ASSERT_EQ(trim_trace_file(m_path.string(),
                              m_trimmed.string(),
                              reader_types::trace_schema_t::rocpd,
                              version_t{ 2, 0, 0 },
                              160,
                              170,
                              nullptr),
              PH_RESULT_SUCCESS);

    reader_t trimmed{ std::make_unique<storage_t>(m_trimmed.string(), "") };
    EXPECT_EQ(trimmed.get_event_count(
                  reader_types::event_filter_t{ .types = { event_type_t::region } }),
              2U);
    EXPECT_EQ(trimmed.get_event_count(reader_types::event_filter_t{
                  .types = { event_type_t::kernel_dispatch } }),
              1U);
    EXPECT_EQ(trimmed.get_event_count(
                  reader_types::event_filter_t{ .types = { event_type_t::memory_copy } }),
              0U);
}
