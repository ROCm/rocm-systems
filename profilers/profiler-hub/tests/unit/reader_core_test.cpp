// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_test_fixture.hpp"

#include "profiler-hub/reader.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace profiler_hub;
using namespace profiler_hub::test;

TEST_F(reader_test, get_all_threads_is_readable_after_flush)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = 1;
    writer->register_process_info(process_info);

    writer_types::thread_info_t thread_info;
    thread_info.thread_id  = 200;
    thread_info.node_id    = 1;
    thread_info.process_id = 100;
    writer->register_thread_info(thread_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader  = make_reader();
    auto threads = reader->get_all_threads();

    ASSERT_EQ(threads.size(), 1);
    EXPECT_EQ(threads[0]->thread_id, 200);
}

TEST_F(reader_test, get_events_for_track_returns_empty_for_null_track)
{
    auto writer = make_writer();
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    EXPECT_TRUE(reader->get_events_for_track(nullptr).empty());
}

// DISABLED: get_tracks() only surfaces tracks with at least one event/sample, so a
// track registered with no events returns 0 tracks here, unlike develop's
// get_all_tracks(), which returned raw registered rocpd_track rows.
TEST_F(reader_test, DISABLED_get_events_for_track_returns_events_for_registered_track)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::track_info_t track_info;
    track_info.node_id = 1;
    writer->register_track_info(track_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto tracks = reader->get_tracks();
    ASSERT_EQ(tracks.size(), 1);

    EXPECT_TRUE(reader->get_events_for_track(tracks[0]).empty());
}

// DISABLED: get_region_details, get_kernel_dispatch_details, get_memory_copy_details,
// get_memory_alloc_details, get_sample_details, and get_pmc_event_details were
// consolidated into get_event_info(event_id_t); they do not exist. Left commented
// out (not DISABLED_-prefixed) because the calls below would not compile.
/*
TEST_F(reader_test, get_region_details_returns_matching_data)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    auto region = reader->get_region_details(events[0]);
    ASSERT_TRUE(region.has_value());
    EXPECT_EQ(region->name, "test-region");
    EXPECT_EQ(region->start_timestamp, 1000);
    EXPECT_EQ(region->end_timestamp, 2000);
}

TEST_F(reader_test, get_kernel_dispatch_details_returns_nullopt_for_region_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_kernel_dispatch_details(events[0]).has_value());
}

TEST_F(reader_test, get_memory_copy_details_returns_nullopt_for_region_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_memory_copy_details(events[0]).has_value());
}

TEST_F(reader_test, get_memory_alloc_details_returns_nullopt_for_region_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_memory_alloc_details(events[0]).has_value());
}

TEST_F(reader_test, get_sample_details_returns_nullopt)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_sample_details(events[0]).has_value());
}

TEST_F(reader_test, get_pmc_event_details_returns_nullopt)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_FALSE(reader->get_pmc_event_details(events[0]).has_value());
}
*/

TEST_F(reader_test, get_arguments_returns_registered_arg)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    auto args = reader->get_arguments(events[0]);
    ASSERT_EQ(args.size(), 1);
    EXPECT_EQ(args[0]->name, "x");
    EXPECT_EQ(args[0]->value, "5");
}

TEST_F(reader_test, get_call_stack_and_source_context_do_not_throw)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_NO_THROW(reader->get_call_stack(events[0]));
    EXPECT_NO_THROW(reader->get_source_context(events[0]));
}

TEST_F(reader_test, get_correlated_events_excludes_self)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto events = reader->get_events();
    ASSERT_EQ(events.size(), 1);

    EXPECT_TRUE(reader->get_correlated_events(events[0]).empty());
}

TEST_F(reader_test, get_kernel_summary_and_region_summary_return_empty)
{
    auto writer = make_writer();
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();

    EXPECT_TRUE(reader->get_kernel_summary().empty());
    EXPECT_TRUE(reader->get_region_summary().empty());
}

TEST_F(reader_test, get_data_time_range_reflects_inserted_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto window = reader->get_time_range();

    ASSERT_TRUE(window.start.has_value());
    ASSERT_TRUE(window.end.has_value());
    EXPECT_EQ(window.start.value(), 1000);
    EXPECT_EQ(window.end.value(), 2000);
}

TEST_F(reader_test, get_event_counts_reflects_inserted_event)
{
    auto writer = make_writer();
    seed_region_with_full_event(*writer);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader = make_reader();
    auto counts = reader->get_event_counts();

    EXPECT_EQ(counts.at(reader_types::event_type_t::region), 1);
}

// ============================================================================
// Track-scoped API tests — v4.0 real-capture fixture (rocpd_v4.db)
// cpu_thread + gpu_queue + dma interval tracks and flows. This fixture has no
// counter samples, so the scalar path is covered by reader_v4_counter_test.
// ============================================================================

class reader_v4_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V4_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

// Asserts the reader selects the v4 read backend on this rocpd table suffix by
// checking a v4-only oracle that a v3 backend cannot reproduce on this capture:
// the cpu_thread interval spine is materialized through rocpd_timestamp
// (v4-only; 384 regions) and the GPU agent resolves to an MI300X. Stream tracks are
// excluded because they exist on both v3 and v4 paths and so cannot discriminate the
// backend. m_is_v4 is private; this is its public behavioral proxy.
TEST_F(reader_v4_test, selects_v4_backend_on_underscore_joined_hyphenated_suffix)
{
    auto tracks = m_reader->get_tracks();

    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto intervals = m_reader->get_interval_track(cpu->id);
    ASSERT_EQ(intervals.size(), 384);
    ASSERT_EQ(intervals.front().start, 516609802359041);

    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    ASSERT_NE(gpu->agent_info, nullptr);
    ASSERT_EQ(gpu->agent_info->name, "AMD Instinct MI300X");
}

TEST_F(reader_v4_test, v4_track_classification_and_identity)
{
    auto tracks = m_reader->get_tracks();
    ASSERT_EQ(tracks.size(), 5);

    auto cpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    auto gpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    auto dma = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(cpu.size(), 1);
    ASSERT_EQ(gpu.size(), 1);
    ASSERT_EQ(dma.size(), 2);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream).size(), 1U);

    // Q10: v4 GPU tracks are scoped to agent.
    ASSERT_NE(gpu[0]->agent_info, nullptr);
    ASSERT_EQ(gpu[0]->agent_info->name, "AMD Instinct MI300X");
    ASSERT_NE(gpu[0]->queue_info, nullptr);
    ASSERT_EQ(gpu[0]->queue_info->name, "Queue 0");

    bool saw_gpu_agent = false, saw_cpu_agent = false;
    for(const auto& d : dma)
    {
        ASSERT_NE(d->agent_info, nullptr);
        if(d->agent_info->agent_type == "GPU") saw_gpu_agent = true;
        if(d->agent_info->agent_type == "CPU") saw_cpu_agent = true;
    }
    ASSERT_TRUE(saw_gpu_agent);
    ASSERT_TRUE(saw_cpu_agent);
}

TEST_F(reader_v4_test, v4_get_interval_track_cpu_thread_regions)
{
    auto tracks = m_reader->get_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto intervals = m_reader->get_interval_track(cpu->id);
    ASSERT_EQ(intervals.size(), 384);
    ASSERT_TRUE(is_start_sorted(intervals));

    const auto& first = intervals.front();
    ASSERT_EQ(first.start, 516609802359041);
    ASSERT_EQ(first.end, 516609802359341);
    ASSERT_GE(first.end, first.start);
    ASSERT_GT(row_id_of(first.id), 0U);

    ASSERT_EQ(type_of(first.id), profiler_hub::reader_types::event_type_t::region);
    ASSERT_TRUE(m_reader->get_event_info(first.id).has_value());
}

TEST_F(reader_v4_test, v4_get_interval_track_cpu_thread_carries_category)
{
    // v4 resolves category through rocpd_info_category, unlike v3's rocpd_string.
    auto tracks = m_reader->get_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto intervals = m_reader->get_interval_track(cpu->id);
    ASSERT_EQ(intervals.size(), 384);

    ASSERT_EQ(intervals.front().category, "hsa_api");
    for(const auto& ev : intervals)
    {
        auto details = m_reader->get_event_info(ev.id);
        ASSERT_TRUE(details.has_value());
        ASSERT_EQ(ev.category, details->category);
        ASSERT_EQ(ev.category, "hsa_api");
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_gpu_queue_dispatches)
{
    auto tracks = m_reader->get_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);

    auto intervals = m_reader->get_interval_track(gpu->id);
    ASSERT_EQ(intervals.size(), 20);
    ASSERT_TRUE(is_start_sorted(intervals));
    ASSERT_EQ(intervals.front().start, 516609921772013);
    ASSERT_EQ(intervals.front().end, 516609921781427);

    ASSERT_EQ(type_of(intervals.front().id),
              profiler_hub::reader_types::event_type_t::kernel_dispatch);
    ASSERT_TRUE(m_reader->get_event_info(intervals.front().id).has_value());
}

TEST_F(reader_v4_test, v4_gpu_queue_track_carries_agent_id)
{
    // agent_id lives on rocpd_track in the v4 schema (same contract as v3).
    auto tracks = m_reader->get_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    ASSERT_NE(gpu->agent_info, nullptr);
    ASSERT_EQ(gpu->agent_info->id, 6);
    ASSERT_EQ(gpu->agent_info->agent_type, "GPU");
}

TEST_F(reader_v4_test, v4_get_interval_track_gpu_queue_carries_category)
{
    // v4 resolves kernel-dispatch category through rocpd_info_category, unlike v3's
    // rocpd_string.
    auto tracks = m_reader->get_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);

    auto intervals = m_reader->get_interval_track(gpu->id);
    ASSERT_EQ(intervals.size(), 20U);

    ASSERT_EQ(intervals.front().category, "kernel_dispatch");
    for(const auto& ev : intervals)
    {
        auto details = m_reader->get_event_info(ev.id);
        ASSERT_TRUE(details.has_value());
        ASSERT_EQ(ev.category, details->category);
        ASSERT_EQ(ev.category, "kernel_dispatch");
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_dma_memory_copies)
{
    auto tracks = m_reader->get_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 2);

    for(const auto& d : dma)
    {
        auto intervals = m_reader->get_interval_track(d->id);
        ASSERT_EQ(intervals.size(), 1);
        ASSERT_GE(intervals.front().end, intervals.front().start);
        ASSERT_EQ(type_of(intervals.front().id),
                  profiler_hub::reader_types::event_type_t::memory_copy);
        ASSERT_TRUE(m_reader->get_event_info(intervals.front().id).has_value());
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_dma_carries_category)
{
    // v4 resolves memory-copy category through rocpd_info_category, unlike v3's
    // rocpd_string.
    auto tracks = m_reader->get_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 2);

    for(const auto& d : dma)
    {
        auto intervals = m_reader->get_interval_track(d->id);
        ASSERT_EQ(intervals.size(), 1U);
        for(const auto& ev : intervals)
        {
            ASSERT_EQ(ev.category, "memory_copy");
            auto details = m_reader->get_event_info(ev.id);
            ASSERT_TRUE(details.has_value());
            ASSERT_EQ(ev.category, details->category);
        }
    }
}

// Closes v4 get_event_info parity with v3: the v4 interval tests above assert only
// .has_value() + .category, while v3 asserts the full unified header (name/ts/te) and
// the typed properties bag. v4 resolves the header + properties through a different
// table set than v3 (rocpd_timestamp spine + rocpd_string + rocpd_info_category). ts/te
// are checked against the interval's own start/end, not fixed constants, so these
// assertions stay valid across fixture edits.
TEST_F(reader_v4_test, v4_get_event_info_region_header)
{
    auto tracks = m_reader->get_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto intervals = m_reader->get_interval_track(cpu->id);
    ASSERT_FALSE(intervals.empty());
    const auto& first = intervals.front();

    auto detail = m_reader->get_event_info(first.id);
    ASSERT_TRUE(detail.has_value());
    // name resolves from rocpd_string; ts/te resolve from the rocpd_timestamp spine.
    EXPECT_EQ(detail->name, "hsa_system_get_major_extension_table");
    EXPECT_EQ(detail->category, first.category);
    EXPECT_EQ(detail->ts, first.start);
    ASSERT_TRUE(detail->te.has_value());
    EXPECT_EQ(detail->te.value(), first.end);
}

TEST_F(reader_v4_test, v4_get_event_info_kernel_dispatch_properties)
{
    auto tracks = m_reader->get_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);

    auto intervals = m_reader->get_interval_track(gpu->id);
    ASSERT_FALSE(intervals.empty());
    const auto& first = intervals.front();
    ASSERT_EQ(type_of(first.id),
              profiler_hub::reader_types::event_type_t::kernel_dispatch);

    auto detail = m_reader->get_event_info(first.id);
    ASSERT_TRUE(detail.has_value());
    EXPECT_EQ(detail->ts, first.start);
    ASSERT_TRUE(detail->te.has_value());
    EXPECT_EQ(detail->te.value(), first.end);

    // Pre-parity, the v4 interval detail arm asserted none of these typed properties.
    auto* dispatch_id = find_prop(*detail, "dispatch_id");
    ASSERT_NE(dispatch_id, nullptr);
    ASSERT_TRUE(std::holds_alternative<uint64_t>(*dispatch_id));
    EXPECT_EQ(std::get<uint64_t>(*dispatch_id), 1U);

    auto* wg_x = find_prop(*detail, "workgroup_size_x");
    ASSERT_NE(wg_x, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*wg_x), 16U);

    auto* grid_x = find_prop(*detail, "grid_size_x");
    ASSERT_NE(grid_x, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*grid_x), 1024U);

    // Linked entity collapsed to an integer id (mirrors the v3 contract).
    EXPECT_NE(find_prop(*detail, "kernel_symbol_id"), nullptr);
}

TEST_F(reader_v4_test, v4_get_event_info_memory_copy_properties)
{
    auto tracks = m_reader->get_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_FALSE(dma.empty());

    std::optional<profiler_hub::reader_types::event_info_t> h2d;
    for(const auto& d : dma)
    {
        auto intervals = m_reader->get_interval_track(d->id);
        ASSERT_EQ(intervals.size(), 1U);
        const auto& iv = intervals.front();
        ASSERT_EQ(type_of(iv.id), profiler_hub::reader_types::event_type_t::memory_copy);
        auto detail = m_reader->get_event_info(iv.id);
        ASSERT_TRUE(detail.has_value());
        EXPECT_EQ(detail->ts, iv.start);
        ASSERT_TRUE(detail->te.has_value());
        EXPECT_EQ(detail->te.value(), iv.end);
        if(detail->name == "MEMORY_COPY_HOST_TO_DEVICE") h2d = detail;
    }

    ASSERT_TRUE(h2d.has_value());
    auto* size = find_prop(*h2d, "size");
    ASSERT_NE(size, nullptr);
    ASSERT_TRUE(std::holds_alternative<uint64_t>(*size));
    EXPECT_EQ(std::get<uint64_t>(*size), 4194304U);

    EXPECT_NE(find_prop(*h2d, "src_agent_id"), nullptr);
    EXPECT_NE(find_prop(*h2d, "dst_agent_id"), nullptr);
}

TEST_F(reader_v4_test, v4_get_scalar_track_on_interval_track_returns_empty)
{
    // Q7.
    auto tracks = m_reader->get_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    ASSERT_TRUE(m_reader->get_scalar_track(gpu->id).empty());
}

TEST_F(reader_v4_test, v4_get_flows_links_regions_to_gpu_events)
{
    // Flat clique, so the new v4 categories add nothing here; this asserts type-tag
    // parity with the v3 backend (every source is a region; dest is GPU-side).
    using fk   = profiler_hub::reader_types::flow_kind_t;
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 22);
    for(const auto& f : flows)
    {
        ASSERT_GT(row_id_of(f.source), 0U);
        ASSERT_GT(row_id_of(f.dest), 0U);
        ASSERT_EQ(type_of(f.source), profiler_hub::reader_types::event_type_t::region);
        ASSERT_TRUE(m_reader->get_event_info(f.source).has_value());
        ASSERT_EQ(count_interval_resolutions(*m_reader, f.dest), 1);
        ASSERT_NE(type_of(f.dest), profiler_hub::reader_types::event_type_t::region);
        // flow_id is the source's stack_id (directed/typed parity with v3).
        ASSERT_GT(flow_id_value(f.flow_id), 0U);
        ASSERT_TRUE(f.kind == fk::launch_to_dispatch ||
                    f.kind == fk::copy_submit_to_exec);
    }
}

TEST_F(reader_v4_test, v4_get_track_stats_matches_slices_for_interval_tracks)
{
    // v4 stats resolve MIN/MAX through the timestamp spine (start_id/end_id ->
    // rocpd_timestamp).
    auto tracks = m_reader->get_tracks();

    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    auto cpu_intervals = m_reader->get_interval_track(cpu->id);
    auto cpu_stats     = m_reader->get_track_stats(cpu->id);
    expect_stats_match_intervals(cpu_stats, cpu_intervals);
    ASSERT_EQ(cpu_stats.count, 384U);
    ASSERT_EQ(cpu_stats.min_ts.value(), 516609802359041U);

    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    auto gpu_intervals = m_reader->get_interval_track(gpu->id);
    auto gpu_stats     = m_reader->get_track_stats(gpu->id);
    expect_stats_match_intervals(gpu_stats, gpu_intervals);
    ASSERT_EQ(gpu_stats.count, 20U);
    ASSERT_EQ(gpu_stats.min_ts.value(), 516609921772013U);

    for(const auto& d :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma))
    {
        auto intervals = m_reader->get_interval_track(d->id);
        expect_stats_match_intervals(m_reader->get_track_stats(d->id), intervals);
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_stream_aggregates_ops_with_op_kind)
{
    // v4 stream tracks synthesize from DISTINCT (nid,pid,stream_id) on rocpd_track;
    // each UNION leg joins on stream_id and resolves times through the timestamp
    // spine. The stream aggregates across ops, so its earliest start (a memory_copy)
    // precedes the gpu_queue's first dispatch -- proof the stream is not just the
    // queue track relabeled. Handles classify via type_of() and resolve through
    // get_event_info (op_kind is retired).
    auto tracks = m_reader->get_tracks();
    auto stream =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_NE(stream, nullptr);

    auto intervals = m_reader->get_interval_track(stream->id);
    ASSERT_EQ(intervals.size(), 22U);
    ASSERT_TRUE(is_start_sorted(intervals));
    ASSERT_EQ(intervals.front().start, 516609915990946);

    size_t kd = 0, mc = 0;
    for(const auto& ev : intervals)
    {
        ASSERT_GE(ev.end, ev.start);
        ASSERT_EQ(count_interval_resolutions(*m_reader, ev.id), 1)
            << "handle must resolve through exactly one detail accessor";
        if(type_of(ev.id) == profiler_hub::reader_types::event_type_t::kernel_dispatch)
            ++kd;
        else if(type_of(ev.id) == profiler_hub::reader_types::event_type_t::memory_copy)
            ++mc;
        else
            FAIL() << "unexpected event type on stream 0";
    }
    ASSERT_EQ(kd, 20U);
    ASSERT_EQ(mc, 2U);
}

TEST_F(reader_v4_test, v4_get_track_stats_stream_matches_interval_slice)
{
    auto tracks = m_reader->get_tracks();
    auto stream =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_NE(stream, nullptr);

    auto intervals = m_reader->get_interval_track(stream->id);
    auto stats     = m_reader->get_track_stats(stream->id);
    expect_stats_match_intervals(stats, intervals);
    ASSERT_EQ(stats.count, 22U);
}

// The windowed-count path has a separate v4 SQL implementation
// (read_statements_v4.hpp).
TEST_F(reader_v4_test, v4_get_event_counts_time_window_filters)
{
    using event_type_t    = profiler_hub::reader_types::event_type_t;
    const auto all_events = m_reader->get_events();
    ASSERT_GE(all_events.size(), 2U);

    uint64_t min_start = all_events.front().start_timestamp;
    uint64_t max_start = all_events.front().start_timestamp;
    for(const auto& e : all_events)
    {
        min_start = std::min<uint64_t>(min_start, e.start_timestamp);
        max_start = std::max<uint64_t>(max_start, e.start_timestamp);
    }
    ASSERT_LT(min_start, max_start);
    profiler_hub::reader_types::time_window_t window;
    window.start = min_start;
    window.end   = min_start + (max_start - min_start) / 2;

    const auto unwindowed = m_reader->get_event_counts();
    const auto windowed   = m_reader->get_event_counts(window);

    profiler_hub::reader_types::event_filter_t wfilter;
    wfilter.time_window                            = window;
    const auto                     windowed_events = m_reader->get_events(wfilter);
    std::map<event_type_t, size_t> per_type;
    for(const auto& e : windowed_events)
        per_type[e.unique_identifier.type]++;

    size_t unwindowed_total = 0;
    size_t windowed_total   = 0;
    for(auto t : { event_type_t::region,
                   event_type_t::kernel_dispatch,
                   event_type_t::memory_copy,
                   event_type_t::memory_allocate })
    {
        ASSERT_EQ(windowed.at(t), per_type[t]);
        ASSERT_LE(windowed.at(t), unwindowed.at(t));
        unwindowed_total += unwindowed.at(t);
        windowed_total += windowed.at(t);
    }
    ASSERT_LT(windowed_total, unwindowed_total);
}
