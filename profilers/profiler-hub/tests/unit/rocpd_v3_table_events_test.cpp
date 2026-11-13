// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_query.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "rocpd_v2_fixture.hpp"
#include "trace_fixtures.hpp"

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{

using namespace profiler_hub;
using reader_types::track_kind_t;

const ph_table_event*
find_row(const std::vector<ph_table_event>& rows, uint64_t start)
{
    for(const auto& row : rows)
    {
        if(row.start == start) return &row;
    }
    return nullptr;
}

const name_duration_summary_t*
find_summary(const std::vector<name_duration_summary_t>& rows, const std::string& name)
{
    for(const auto& row : rows)
    {
        if(row.name == name) return &row;
    }
    return nullptr;
}

class rocpd_v3_table_events_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_path = std::filesystem::temp_directory_path() / "rocpd_v3_table_events_test.db";
        std::filesystem::remove(m_path);
        write_trace();
        attach_counter_to_kernel();
        add_hip_event();
        m_reader =
            std::make_unique<reader_t>(std::make_unique<storage_t>(m_path.string(), ""));
    }

    void TearDown() override
    {
        m_reader.reset();
        std::filesystem::remove(m_path);
    }

    void write_trace() const
    {
        auto writer = test::make_trace_writer(m_path.string());
        test::register_thread_track(*writer);

        writer_types::agent_info_t gpu;
        gpu.unique_id.agent_type = "GPU";
        gpu.unique_id.type_index = 0;
        gpu.node_id              = 1;
        gpu.process_id           = 100;
        writer->register_agent_info(gpu);

        writer_types::agent_info_t other;
        other.unique_id.agent_type = "GPU";
        other.unique_id.type_index = 1;
        other.node_id              = 1;
        other.process_id           = 100;
        writer->register_agent_info(other);

        writer_types::queue_info_t queue;
        queue.queue_id   = 30;
        queue.node_id    = 1;
        queue.process_id = 100;
        writer->register_queue_info(queue);

        writer_types::stream_info_t stream;
        stream.stream_id  = 40;
        stream.node_id    = 1;
        stream.process_id = 100;
        writer->register_stream_info(stream);

        writer_types::code_object_info_t code_object;
        code_object.id         = 10;
        code_object.node_id    = 1;
        code_object.process_id = 100;
        writer->register_code_object_info(code_object);

        auto add_symbol = [&](size_t id, const char* name, const char* display) {
            writer_types::kernel_symbol_info_t symbol;
            symbol.id                   = id;
            symbol.node_id              = 1;
            symbol.process_id           = 100;
            symbol.code_obj_id          = code_object.id;
            symbol.group_segment_size   = 128;
            symbol.private_segment_size = 32;
            if(name != nullptr) symbol.name = name;
            if(display != nullptr) symbol.display_name = display;
            writer->register_kernel_symbol_info(symbol);
        };
        add_symbol(20, "vector_add_internal", "vector_add");
        add_symbol(21, "plain_kernel", nullptr);
        add_symbol(22, nullptr, nullptr);

        writer_types::track_info_t markers;
        markers.name       = "markers";
        markers.node_id    = 1;
        markers.process_id = 100;
        markers.thread_id  = 200;
        writer->register_track_info(markers);

        writer_types::pmc_info_t pmc;
        pmc.unique_id.name     = "SCLK";
        pmc.unique_id.agent_id = gpu.unique_id;
        pmc.symbol             = "SCLK";
        pmc.node_id            = 1;
        pmc.process_id         = 100;
        writer->register_pmc_info(pmc);

        writer_types::track_info_t counter_track;
        counter_track.name       = "counter-track";
        counter_track.node_id    = 1;
        counter_track.process_id = 100;
        writer->register_track_info(counter_track);

        writer_types::trace_environment_t host;
        host.node_id    = 1;
        host.process_id = 100;
        host.thread_id  = 200;

        // The first event written by this writer is stored with event id 0, and
        // the table query's inner join does not find that row. A discarded
        // region takes that id so the rows below are readable.
        test::insert_region(*writer, "region-z", 100, 200);

        writer_types::event_data_t categorized;
        categorized.event_category = "HIP_API";
        writer_types::region_data_t region_a;
        region_a.name            = "region-a";
        region_a.start_timestamp = 1000;
        region_a.end_timestamp   = 2000;
        region_a.event           = categorized;
        writer->insert_region_data(region_a, host);

        test::insert_region(*writer, "region-b", 5000, 9000);

        host.track_name = "markers";
        writer_types::region_data_t region_s;
        region_s.name            = "region-s";
        region_s.start_timestamp = 1500;
        region_s.end_timestamp   = 1600;
        region_s.event           = writer_types::event_data_t{};
        writer->insert_region_data(region_s, host);
        host.track_name.reset();

        writer_types::trace_environment_t device = host;
        device.agent_id                          = gpu.unique_id;
        device.queue_id                          = queue.queue_id;
        device.stream_id                         = stream.stream_id;

        auto add_kernel = [&](size_t symbol, uint64_t start, uint64_t end, size_t grid) {
            writer_types::kernel_dispatch_data_t kernel;
            kernel.dispatch_id          = symbol;
            kernel.kernel_symbol_id     = symbol;
            kernel.start_timestamp      = start;
            kernel.end_timestamp        = end;
            kernel.grid_size_x          = grid;
            kernel.grid_size_y          = 4;
            kernel.grid_size_z          = 2;
            kernel.workgroup_size_x     = 64;
            kernel.workgroup_size_y     = 2;
            kernel.workgroup_size_z     = 1;
            kernel.group_segment_size   = 256;
            kernel.private_segment_size = 16;
            kernel.event                = writer_types::event_data_t{};
            writer->insert_kernel_dispatch_data(kernel, device);
        };
        add_kernel(20, 50, 60, 1);
        add_kernel(20, 3000, 3500, 8);
        add_kernel(21, 4000, 4600, 1);
        add_kernel(22, 4700, 4800, 1);

        writer_types::memory_copy_data_t copy;
        copy.name            = "copy-c";
        copy.region_name     = "copy-region";
        copy.start_timestamp = 7000;
        copy.end_timestamp   = 7500;
        copy.size            = 64;
        copy.dst_agent_id    = gpu.unique_id;
        copy.src_agent_id    = other.unique_id;
        copy.dst_address     = 0x1000;
        copy.src_address     = 0x2000;
        copy.event           = writer_types::event_data_t{};
        copy.start_timestamp = 10;
        copy.end_timestamp   = 20;
        writer->insert_memory_copy_data(copy, device);
        copy.start_timestamp = 7000;
        copy.end_timestamp   = 7500;
        writer->insert_memory_copy_data(copy, device);

        auto add_alloc = [&](const char*                       type,
                             uint64_t                          start,
                             size_t                            address,
                             const writer_types::agent_info_t& agent) {
            writer_types::trace_environment_t where = device;
            where.agent_id                          = agent.unique_id;
            writer_types::memory_alloc_data_t alloc;
            alloc.type            = type;
            alloc.level           = "REAL";
            alloc.start_timestamp = start;
            alloc.end_timestamp   = start + 100;
            alloc.address         = address;
            alloc.size            = 128;
            alloc.event           = writer_types::event_data_t{};
            writer->insert_memory_alloc_data(alloc, where);
        };
        add_alloc("ALLOC", 30, 0x10, gpu);
        add_alloc("ALLOC", 8000, 0x3000, gpu);
        add_alloc("FREE", 8200, 0x3000, gpu);
        add_alloc("ALLOC", 9000, 0x4000, other);

        writer_types::pmc_event_data_t unused;
        unused.event.emplace();
        unused.value            = 9;
        unused.sample.timestamp = 10;
        unused.sample.track     = counter_track;
        writer->insert_pmc_event_data(unused, pmc.unique_id);

        for(const double value : { 1.5, 2.5 })
        {
            writer_types::pmc_event_data_t sample;
            sample.event.emplace();
            sample.value            = value;
            sample.sample.timestamp = value == 1.5 ? 1000 : 2000;
            sample.sample.track     = counter_track;
            writer->insert_pmc_event_data(sample, pmc.unique_id);
        }
        writer->flush_in_memory_data_to_disk();
    }

    void attach_counter_to_kernel() const
    {
        const std::string uuid{ test::trace_uuid };
        sqlite3*          db = nullptr;
        ASSERT_EQ(sqlite3_open(m_path.string().c_str(), &db), SQLITE_OK);
        try
        {
            const std::string sql =
                "UPDATE rocpd_pmc_event_" + uuid +
                " SET event_id = (SELECT event_id FROM rocpd_kernel_dispatch_" + uuid +
                " WHERE start = 3000) WHERE value = 1.5";
            test::exec_sql(db, sql.c_str());
        } catch(...)
        {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);
    }

    void add_hip_event() const
    {
        const std::string uuid{ test::trace_uuid };
        sqlite3*          db = nullptr;
        ASSERT_EQ(sqlite3_open(m_path.string().c_str(), &db), SQLITE_OK);
        try
        {
            const std::string sql =
                "CREATE TABLE rocpd_hip_event_" + uuid +
                "(id INTEGER, start INTEGER, \"end\" INTEGER, name_id INTEGER, "
                "event_id INTEGER, nid INTEGER, pid INTEGER, tid INTEGER, agent_id "
                "INTEGER, queue_id INTEGER, stream_id INTEGER, hip_event_handle INTEGER, "
                "source_queue_id INTEGER);"
                "INSERT INTO rocpd_hip_event_" +
                uuid +
                " SELECT 1, 4100, 4200, NULL, event_id, nid, pid, tid, agent_id, "
                "queue_id, stream_id, 7, queue_id FROM rocpd_kernel_dispatch_" +
                uuid + " ORDER BY id LIMIT 1;";
            test::exec_sql(db, sql.c_str());
        } catch(...)
        {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);
    }

    [[nodiscard]] std::vector<ph_table_event> read(const resolved_event_query& query,
                                                   uint32_t                    type,
                                                   uint64_t                    after = 0,
                                                   uint64_t last     = UINT64_MAX,
                                                   uint32_t capacity = 32) const
    {
        std::vector<ph_table_event> rows(capacity);
        const uint32_t              count =
            m_reader->read_table_events(query, type, after, last, rows.data(), capacity);
        rows.resize(count);
        return rows;
    }

    [[nodiscard]] std::vector<uint32_t> track_ids(track_kind_t kind) const
    {
        std::vector<uint32_t> ids;
        for(const auto& track : m_reader->get_all_tracks())
        {
            if(track && track->category == kind)
            {
                ids.push_back(static_cast<uint32_t>(track->id));
            }
        }
        return ids;
    }

    std::filesystem::path     m_path;
    std::unique_ptr<reader_t> m_reader;
};

TEST_F(rocpd_v3_table_events_test, duration_tables_report_fields_spans_and_summaries)
{
    const auto regions = read(resolved_event_query{}, PH_EVENT_TYPE_REGION);
    ASSERT_EQ(regions.size(), 2U);
    const auto* region_a = find_row(regions, 1000);
    ASSERT_NE(region_a, nullptr);
    EXPECT_EQ(region_a->end, 2000U);
    EXPECT_EQ(region_a->pid, 100U);
    EXPECT_EQ(region_a->tid, 200U);
    EXPECT_NE(region_a->name_id, 0U);
    EXPECT_NE(region_a->category_id, 0U);

    const auto samples = read(resolved_event_query{}, PH_EVENT_TYPE_REGION_SAMPLE);
    ASSERT_EQ(samples.size(), 1U);
    EXPECT_EQ(samples[0].start, 1500U);

    const auto  kernels = read(resolved_event_query{}, PH_EVENT_TYPE_KERNEL_DISPATCH);
    std::string kernel_starts;
    for(const auto& row : kernels)
    {
        kernel_starts += std::to_string(row.start) + " ";
    }
    ASSERT_EQ(kernels.size(), 3U) << kernel_starts;
    const auto* kernel = find_row(kernels, 3000);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->grid_size_x, 8U);
    EXPECT_EQ(kernel->grid_size_y, 4U);
    EXPECT_EQ(kernel->workgroup_size_x, 64U);
    EXPECT_EQ(kernel->group_segment_size, 256U);
    EXPECT_EQ(kernel->private_segment_size, 16U);
    EXPECT_EQ(kernel->static_group_segment_size, 128U);
    EXPECT_EQ(kernel->static_private_segment_size, 32U);
    EXPECT_NE(kernel->kernel_id, 0U);

    const auto page = read(
        resolved_event_query{}, PH_EVENT_TYPE_KERNEL_DISPATCH, kernel->id, UINT64_MAX, 1);
    ASSERT_EQ(page.size(), 1U);
    EXPECT_GT(page[0].id, kernel->id);

    const auto  copies = read(resolved_event_query{}, PH_EVENT_TYPE_MEMORY_COPY);
    const auto* copy   = find_row(copies, 7000);
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->size, 64U);
    EXPECT_EQ(copy->address, 0x1000U);
    EXPECT_EQ(copy->src_address, 0x2000U);
    EXPECT_NE(copy->src_agent_id, 0U);

    const auto allocs = read(resolved_event_query{}, PH_EVENT_TYPE_MEMORY_ALLOCATE);
    ASSERT_NE(find_row(allocs, 8000), nullptr);
    ASSERT_NE(find_row(allocs, 8200), nullptr);
    ASSERT_NE(find_row(allocs, 9000), nullptr);
    const auto* allocation = find_row(allocs, 8000);
    ASSERT_NE(allocation, nullptr);
    EXPECT_STREQ(allocation->mem_type, "ALLOC");
    EXPECT_STREQ(allocation->level, "REAL");
    EXPECT_EQ(allocation->size, 128U);

    uint64_t first = 0;
    uint64_t last  = 0;
    EXPECT_TRUE(m_reader->table_event_id_span(PH_EVENT_TYPE_REGION, first, last));
    EXPECT_LE(first, last);
    EXPECT_TRUE(
        m_reader->table_event_id_span(PH_EVENT_TYPE_KERNEL_DISPATCH, first, last));
    EXPECT_TRUE(m_reader->table_event_id_span(PH_EVENT_TYPE_MEMORY_COPY, first, last));
    EXPECT_TRUE(
        m_reader->table_event_id_span(PH_EVENT_TYPE_MEMORY_ALLOCATE, first, last));
    EXPECT_TRUE(m_reader->table_event_id_span(PH_EVENT_TYPE_HIP_EVENT, first, last));

    std::vector<name_duration_summary_t> summaries;
    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_REGION, summaries));
    const auto* summary_a = find_summary(summaries, "region-a");
    const auto* summary_b = find_summary(summaries, "region-b");
    ASSERT_NE(summary_a, nullptr);
    ASSERT_NE(summary_b, nullptr);
    EXPECT_EQ(summary_a->count, 1U);
    EXPECT_EQ(summary_a->total_duration, 1000U);
    EXPECT_EQ(summary_b->total_duration, 4000U);

    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_REGION_SAMPLE, summaries));
    ASSERT_NE(find_summary(summaries, "region-s"), nullptr);

    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_KERNEL_DISPATCH, summaries));
    EXPECT_NE(find_summary(summaries, "vector_add"), nullptr);
    EXPECT_NE(find_summary(summaries, "plain_kernel"), nullptr);
    EXPECT_NE(find_summary(summaries, std::to_string(find_row(kernels, 4700)->kernel_id)),
              nullptr);

    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_MEMORY_COPY, summaries));
    EXPECT_NE(find_summary(summaries, "copy-c"), nullptr);

    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_MEMORY_ALLOCATE, summaries));
    const auto* alloc_summary = find_summary(summaries, "ALLOC");
    ASSERT_NE(alloc_summary, nullptr);
    EXPECT_EQ(alloc_summary->count, 3U);
    EXPECT_NE(find_summary(summaries, "FREE"), nullptr);
}

TEST_F(rocpd_v3_table_events_test, filters_narrow_duration_rows)
{
    const auto regions = read(resolved_event_query{}, PH_EVENT_TYPE_REGION);
    const auto kernels = read(resolved_event_query{}, PH_EVENT_TYPE_KERNEL_DISPATCH);
    ASSERT_FALSE(regions.empty());
    ASSERT_FALSE(kernels.empty());

    resolved_event_query window;
    window.time         = time_window{ 1000, 2500 };
    const auto windowed = read(window, PH_EVENT_TYPE_REGION);
    ASSERT_EQ(windowed.size(), 1U);
    EXPECT_EQ(windowed[0].start, 1000U);

    resolved_event_query node;
    node.node_id = regions[0].nid;
    EXPECT_EQ(read(node, PH_EVENT_TYPE_REGION).size(), regions.size());
    node.node_id = 999;
    EXPECT_TRUE(read(node, PH_EVENT_TYPE_REGION).empty());

    resolved_event_query process;
    process.process_id = regions[0].process_id;
    EXPECT_FALSE(read(process, PH_EVENT_TYPE_REGION).empty());
    process.process_id = 999;
    EXPECT_TRUE(read(process, PH_EVENT_TYPE_REGION).empty());

    resolved_event_query thread;
    thread.thread_id = regions[0].thread_id;
    EXPECT_FALSE(read(thread, PH_EVENT_TYPE_REGION).empty());
    thread.thread_id = 999;
    EXPECT_TRUE(read(thread, PH_EVENT_TYPE_REGION).empty());

    resolved_event_query agent;
    agent.agent_id = kernels[0].agent_id;
    EXPECT_FALSE(read(agent, PH_EVENT_TYPE_KERNEL_DISPATCH).empty());
    EXPECT_TRUE(read(agent, PH_EVENT_TYPE_REGION).empty());
    agent.agent_id = 999;
    EXPECT_TRUE(read(agent, PH_EVENT_TYPE_KERNEL_DISPATCH).empty());

    resolved_event_query text;
    text.has_string_filter = true;
    text.string_ids        = { regions[0].name_id, regions[1].name_id };
    EXPECT_EQ(read(text, PH_EVENT_TYPE_REGION).size(), 2U);
    text.include_category = true;
    text.string_ids       = { regions[0].category_id };
    EXPECT_FALSE(read(text, PH_EVENT_TYPE_REGION).empty());

    resolved_event_query kernel_name;
    kernel_name.has_string_filter = true;
    kernel_name.kernel_symbol_ids = { kernels[0].kernel_id };
    const auto named              = read(kernel_name, PH_EVENT_TYPE_KERNEL_DISPATCH);
    ASSERT_EQ(named.size(), 1U);
    EXPECT_EQ(named[0].kernel_id, kernels[0].kernel_id);

    resolved_event_query copy_name;
    copy_name.has_string_filter = true;
    const auto copies           = read(resolved_event_query{}, PH_EVENT_TYPE_MEMORY_COPY);
    ASSERT_FALSE(copies.empty());
    copy_name.string_ids = { copies[0].name_id };
    EXPECT_NE(find_row(read(copy_name, PH_EVENT_TYPE_MEMORY_COPY), 7000), nullptr);

    resolved_event_query no_text;
    no_text.has_string_filter = true;
    EXPECT_TRUE(read(no_text, PH_EVENT_TYPE_REGION).empty());
    EXPECT_TRUE(read(no_text, PH_EVENT_TYPE_MEMORY_ALLOCATE).empty());

    const auto threads = track_ids(track_kind_t::thread);
    const auto streams = track_ids(track_kind_t::stream);
    const auto queues  = track_ids(track_kind_t::kernel_dispatch_agent_queue);
    const auto marked  = track_ids(track_kind_t::thread_sample);
    ASSERT_FALSE(threads.empty());
    ASSERT_FALSE(streams.empty());
    ASSERT_FALSE(queues.empty());
    ASSERT_FALSE(marked.empty());

    resolved_event_query by_thread;
    by_thread.tracks = threads;
    EXPECT_FALSE(read(by_thread, PH_EVENT_TYPE_REGION).empty());
    EXPECT_FALSE(read(by_thread, PH_EVENT_TYPE_KERNEL_DISPATCH).empty());

    resolved_event_query by_stream;
    by_stream.tracks = { streams.front(), queues.front() };
    EXPECT_FALSE(read(by_stream, PH_EVENT_TYPE_KERNEL_DISPATCH).empty());

    resolved_event_query by_marker;
    by_marker.tracks = marked;
    EXPECT_FALSE(read(by_marker, PH_EVENT_TYPE_REGION_SAMPLE).empty());

    resolved_event_query missing;
    missing.tracks = { 999999 };
    EXPECT_TRUE(read(missing, PH_EVENT_TYPE_REGION).empty());

    resolved_event_query typed;
    typed.event_types = { PH_EVENT_TYPE_REGION };
    EXPECT_TRUE(read(typed, PH_EVENT_TYPE_KERNEL_DISPATCH).empty());
    EXPECT_FALSE(read(typed, PH_EVENT_TYPE_REGION).empty());

    EXPECT_EQ(
        m_reader->read_table_events(
            resolved_event_query{}, PH_EVENT_TYPE_REGION, 0, UINT64_MAX, nullptr, 4),
        0U);
    ph_table_event one{};
    EXPECT_EQ(m_reader->read_table_events(
                  resolved_event_query{}, PH_EVENT_TYPE_REGION, 0, UINT64_MAX, &one, 0),
              0U);
    EXPECT_EQ(
        m_reader->read_table_events(resolved_event_query{},
                                    static_cast<uint32_t>(PH_EVENT_TYPE_HIP_EVENT) + 50,
                                    0,
                                    UINT64_MAX,
                                    &one,
                                    1),
        0U);
    EXPECT_TRUE(read(resolved_event_query{}, PH_EVENT_TYPE_REGION, UINT64_MAX).empty());

    std::vector<name_duration_summary_t> summaries;
    EXPECT_TRUE(m_reader->summarize_name_durations(
        typed, PH_EVENT_TYPE_KERNEL_DISPATCH, summaries));
    EXPECT_TRUE(summaries.empty());
    EXPECT_TRUE(
        m_reader->summarize_name_durations(no_text, PH_EVENT_TYPE_REGION, summaries));
    EXPECT_TRUE(summaries.empty());
    EXPECT_TRUE(
        m_reader->summarize_name_durations(missing, PH_EVENT_TYPE_REGION, summaries));
    EXPECT_TRUE(summaries.empty());
    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{},
        static_cast<uint32_t>(PH_EVENT_TYPE_HIP_EVENT) + 50,
        summaries));
}

TEST_F(rocpd_v3_table_events_test, samples_counters_memory_activity_and_hip_events)
{
    const auto samples = read(resolved_event_query{}, PH_EVENT_TYPE_PMC_SAMPLE);
    ASSERT_NE(find_row(samples, 2000), nullptr);
    EXPECT_DOUBLE_EQ(find_row(samples, 2000)->value, 2.5);

    resolved_event_query late;
    late.time = time_window{ 1500, 2500 };
    ASSERT_EQ(read(late, PH_EVENT_TYPE_PMC_SAMPLE).size(), 1U);

    const auto counters_tracks = track_ids(track_kind_t::pmc_agent);
    ASSERT_FALSE(counters_tracks.empty());
    resolved_event_query counter_track;
    counter_track.tracks = counters_tracks;
    EXPECT_FALSE(read(counter_track, PH_EVENT_TYPE_PMC_SAMPLE).empty());
    counter_track.agent_id = 999;
    EXPECT_TRUE(read(counter_track, PH_EVENT_TYPE_PMC_SAMPLE).empty());

    const auto kernel_counters = read(resolved_event_query{}, PH_EVENT_TYPE_PMC_COUNTER);
    ASSERT_EQ(kernel_counters.size(), 1U);
    EXPECT_EQ(kernel_counters[0].start, 3000U);
    EXPECT_DOUBLE_EQ(kernel_counters[0].value, 1.5);

    resolved_event_query counter_window;
    counter_window.time = time_window{ 3000, 3500 };
    EXPECT_EQ(read(counter_window, PH_EVENT_TYPE_PMC_COUNTER).size(), 1U);
    counter_window.time = time_window{ 1, 2 };
    EXPECT_TRUE(read(counter_window, PH_EVENT_TYPE_PMC_COUNTER).empty());

    const auto kernel_tracks = track_ids(track_kind_t::pmc_kernel);
    ASSERT_FALSE(kernel_tracks.empty());
    resolved_event_query by_kernel_counter;
    by_kernel_counter.tracks   = kernel_tracks;
    by_kernel_counter.agent_id = kernel_counters[0].agent_id;
    EXPECT_EQ(read(by_kernel_counter, PH_EVENT_TYPE_PMC_COUNTER).size(), 1U);
    by_kernel_counter.tracks = track_ids(track_kind_t::thread);
    EXPECT_TRUE(read(by_kernel_counter, PH_EVENT_TYPE_PMC_COUNTER).empty());

    const auto memory_tracks = track_ids(track_kind_t::pmc_memory);
    ASSERT_GE(memory_tracks.size(), 2U);
    resolved_event_query memory;
    memory.tracks       = memory_tracks;
    const auto activity = read(memory, PH_EVENT_TYPE_MEMORY_ACTIVITY);
    ASSERT_EQ(activity.size(), 3U);
    EXPECT_LT(activity[0].id, activity[1].id);
    EXPECT_LT(activity[1].id, activity[2].id);

    memory.time = time_window{ 8000, 8300 };
    EXPECT_EQ(read(memory, PH_EVENT_TYPE_MEMORY_ACTIVITY).size(), 2U);

    memory.time.reset();
    memory.agent_id = 999;
    EXPECT_TRUE(read(memory, PH_EVENT_TYPE_MEMORY_ACTIVITY).empty());
    memory.tracks = track_ids(track_kind_t::thread);
    memory.agent_id.reset();
    EXPECT_TRUE(read(memory, PH_EVENT_TYPE_MEMORY_ACTIVITY).empty());

    uint64_t first = 0;
    uint64_t last  = 0;
    EXPECT_TRUE(m_reader->table_event_id_span(PH_EVENT_TYPE_PMC_SAMPLE, first, last));
    EXPECT_LE(first, last);
    EXPECT_TRUE(m_reader->table_event_id_span(PH_EVENT_TYPE_PMC_COUNTER, first, last));
    EXPECT_TRUE(
        m_reader->table_event_id_span(PH_EVENT_TYPE_MEMORY_ACTIVITY, first, last));

    const auto hip = read(resolved_event_query{}, PH_EVENT_TYPE_HIP_EVENT);
    ASSERT_EQ(hip.size(), 1U);
    EXPECT_EQ(hip[0].start, 4100U);
    EXPECT_EQ(hip[0].size, 7U);

    const auto hip_tracks = track_ids(track_kind_t::hip_event_agent_queue);
    ASSERT_FALSE(hip_tracks.empty());
    resolved_event_query hip_query;
    hip_query.tracks = hip_tracks;
    EXPECT_EQ(read(hip_query, PH_EVENT_TYPE_HIP_EVENT).size(), 1U);

    std::vector<name_duration_summary_t> summaries;
    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_PMC_SAMPLE, summaries));
    EXPECT_TRUE(summaries.empty());
    EXPECT_TRUE(m_reader->summarize_name_durations(
        resolved_event_query{}, PH_EVENT_TYPE_HIP_EVENT, summaries));
    EXPECT_FALSE(summaries.empty());
}

}  // namespace
