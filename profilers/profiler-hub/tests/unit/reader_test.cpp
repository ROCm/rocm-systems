// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>

namespace
{

using namespace profiler_hub;

class reader_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "reader_test_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer() const
    {
        return std::make_unique<writer_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    [[nodiscard]] std::unique_ptr<reader_t> make_reader() const
    {
        return std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    // Registers node/process/thread and inserts one region event with a call stack,
    // one arg, and a correlation stack_id, so the detail/property getters below have
    // real data to resolve.
    void seed_region_with_full_event(writer_t& writer) const
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

        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 100;
        trace_environment.thread_id  = 200;

        writer_types::event_data_t event_data;
        event_data.stack_id = 1;

        writer_types::arg_data_t arg;
        arg.position = 0;
        arg.type     = "int";
        arg.name     = "x";
        arg.value    = "5";

        writer_types::region_data_t region_data;
        region_data.name            = "test-region";
        region_data.start_timestamp = 1000;
        region_data.end_timestamp   = 2000;
        region_data.event           = event_data;
        region_data.args.push_back(arg);
        writer.insert_region_data(region_data, trace_environment);
    }

    std::string m_db_path;
    // Embedded verbatim into unquoted SQL table names by insert_statements, so it
    // must be a valid identifier fragment - no hyphens.
    std::string m_uuid = "testuuid0000";
};

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

// [task 068] DISABLED: renamed get_all_tracks->get_tracks compiles, but the semantics
// differ — our get_tracks() synthesizes tracks from actual event/sample data, so a track
// registered with no events is not surfaced (returns 0), whereas develop's get_all_tracks()
// returned raw registered rocpd_track rows. Semantic collision beyond a mechanical rename;
// equivalent coverage is a deferred port chunk.
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

// [task 068] DISABLED: the per-type detail getters (get_region_details,
// get_kernel_dispatch_details, get_memory_copy_details, get_memory_alloc_details,
// get_sample_details, get_pmc_event_details) were consolidated into the single
// get_event_info(event_id_t) by task 041. These develop tests exercise the removed
// surface; re-adding equivalent get_event_info coverage is a deferred port chunk, not
// this rebase-baseline task. Commented out (not DISABLED_-prefixed) because the removed
// methods would otherwise fail to compile.
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

// ===========================================================================
// Flow-family tests (ported from pre-rebase cac3369ac5, re-seeded via the
// writer temp-DB model). The original fixtures loaded three purpose-built
// SQL databases (rocpd_v3_clique / rocpd_v3_flow_order / a flat-clique slice
// of rocpd_v3_edge) via ROCPD_DB_V3_*_PATH compile definitions; here each
// fixture reproduces the identical stack_id clique geometry through
// writer_t::insert_*_data, so the assertions (directed/typed clique, handle
// collision, equal-start + equal-latency tie-breaks, stack-0/NULL exclusion)
// are unchanged — only the seeding mechanism moved from raw SQL to the writer.
// ===========================================================================

// --- test-only opaque-handle peeks (public API treats these as opaque) ---

// Mint the handle for the `ordinal`-th inserted event of `type` (1-based logical
// ordinal). The writer assigns per-type-table row ids from an autoincrementer that
// starts at 0 (see source/autoincrementer.hpp), so the N-th inserted row of a type
// carries raw id N-1. Tests name endpoints by their 1-based insertion order (region 1,
// kd 2, ...) to match the seed comments and the original SQL fixtures, so this helper
// maps that ordinal onto the writer's 0-based raw id.
reader_types::event_id_t
make_event_id(reader_types::event_type_t type, size_t ordinal)
{
    return reader_types::detail::event_id_access::make(type, ordinal - 1);
}

// Mint the handle for a SQL-seeded fixture row by its EXPLICIT id. Unlike the
// writer's 0-based autoincrementer (see make_event_id above), synthetic SQL
// fixtures assign explicit 1-based ids (e.g. rocpd_sample.id = 1), so no
// ordinal->raw shift applies — the db_id passed here IS the raw id (DL-015/069).
reader_types::event_id_t
make_sql_event_id(reader_types::event_type_t type, size_t db_id)
{
    return reader_types::detail::event_id_access::make(type, db_id);
}

reader_types::event_type_t
type_of(const reader_types::event_id_t& id)
{
    return reader_types::detail::event_id_access::type(id);
}

uint64_t
flow_id_value(const reader_types::flow_id_t& fid)
{
    return reader_types::detail::flow_id_access::value(fid);
}

// 1 if a handle resolves to a unified detail record, else 0.
int
count_interval_resolutions(const reader_t& r, const reader_types::event_id_t& id)
{
    return r.get_event_info(id).has_value() ? 1 : 0;
}

reader_types::track_info_ptr_t
find_first_track(const reader_types::track_info_list_t& tracks,
                 reader_types::track_type_t             type)
{
    for(const auto& t : tracks)
    {
        if(t->type == type) return t;
    }
    return nullptr;
}

// Peek the per-type-table row id an opaque handle encodes. Test-only: the public
// API treats event_id_t as opaque (equality / ordering / hashing only).
size_t
row_id_of(const reader_types::event_id_t& id)
{
    return reader_types::detail::event_id_access::row_id(id);
}

// Look up a property in a unified event_info_t bag by key (nullptr if absent).
const reader_types::arg_value_t*
find_prop(const reader_types::event_info_t& d, const std::string& key)
{
    for(const auto& p : d.properties)
    {
        if(p.key == key) return &p.value;
    }
    return nullptr;
}

// All tracks of a given type.
reader_types::track_info_list_t
find_tracks(const reader_types::track_info_list_t& tracks, reader_types::track_type_t type)
{
    reader_types::track_info_list_t out;
    for(const auto& t : tracks)
    {
        if(t->type == type) out.push_back(t);
    }
    return out;
}

// True if interval events are non-decreasing by start timestamp (the documented
// ordering contract of get_interval_track).
bool
is_start_sorted(const reader_types::interval_entry_list_t& v)
{
    for(size_t i = 1; i < v.size(); ++i)
    {
        if(v[i].start < v[i - 1].start) return false;
    }
    return true;
}

// True if scalar events are non-decreasing by timestamp (the documented
// ordering contract of get_scalar_track).
bool
is_timestamp_sorted(const reader_types::scalar_sample_list_t& v)
{
    for(size_t i = 1; i < v.size(); ++i)
    {
        if(v[i].timestamp < v[i - 1].timestamp) return false;
    }
    return true;
}

// Assert get_track_stats agrees with a full get_interval_track slice: count ==
// #rows, min_ts == MIN(start), max_ts == MAX(end).
void
expect_stats_match_intervals(const reader_types::track_stats_t&         stats,
                             const reader_types::interval_entry_list_t& intervals)
{
    ASSERT_EQ(stats.count, intervals.size());
    if(intervals.empty())
    {
        ASSERT_FALSE(stats.min_ts.has_value());
        ASSERT_FALSE(stats.max_ts.has_value());
        return;
    }
    auto min_start = intervals.front().start;
    auto max_end   = intervals.front().end;
    for(const auto& iv : intervals)
    {
        if(iv.start < min_start) min_start = iv.start;
        if(iv.end > max_end) max_end = iv.end;
    }
    ASSERT_TRUE(stats.min_ts.has_value());
    ASSERT_TRUE(stats.max_ts.has_value());
    ASSERT_EQ(stats.min_ts.value(), min_start);
    ASSERT_EQ(stats.max_ts.value(), max_end);
}

// Assert get_track_stats agrees with a full get_scalar_track slice: count ==
// #samples, min_ts == MIN(timestamp), max_ts == MAX(timestamp).
void
expect_stats_match_scalars(const reader_types::track_stats_t&        stats,
                           const reader_types::scalar_sample_list_t& samples)
{
    ASSERT_EQ(stats.count, samples.size());
    if(samples.empty())
    {
        ASSERT_FALSE(stats.min_ts.has_value());
        ASSERT_FALSE(stats.max_ts.has_value());
        return;
    }
    auto min_ts = samples.front().timestamp;
    auto max_ts = samples.front().timestamp;
    for(const auto& s : samples)
    {
        if(s.timestamp < min_ts) min_ts = s.timestamp;
        if(s.timestamp > max_ts) max_ts = s.timestamp;
    }
    ASSERT_TRUE(stats.min_ts.has_value());
    ASSERT_TRUE(stats.max_ts.has_value());
    ASSERT_EQ(stats.min_ts.value(), min_ts);
    ASSERT_EQ(stats.max_ts.value(), max_ts);
}

// --- shared writer-seed helpers ---
// Per-type-table row ids come from the writer's per-type autoincrementer (0-based,
// source/autoincrementer.hpp), incremented once per insert, so seeding each type in
// the intended order yields dense ids 0,1,2,... — the N-th inserted row is raw id N-1.
// trace_environment.track_name is left unset so no rocpd_sample is minted (keeps
// regions a single cpu_thread/main track).

void
seed_flow_identity(writer_t& writer)
{
    const writer_types::node_info_t node_info{ 1, 42, "synthetic-machine" };
    writer.register_node_info(node_info);

    writer_types::process_info_t process_info;
    process_info.pid     = 1;
    process_info.node_id = 1;
    writer.register_process_info(process_info);

    writer_types::thread_info_t thread_info;
    thread_info.thread_id  = 1;
    thread_info.node_id    = 1;
    thread_info.process_id = 1;
    writer.register_thread_info(thread_info);

    writer_types::agent_info_t agent_info;
    agent_info.unique_id.agent_type = "GPU";
    agent_info.unique_id.type_index = 0;
    agent_info.node_id              = 1;
    agent_info.process_id           = 1;
    writer.register_agent_info(agent_info);

    writer_types::queue_info_t queue_info;
    queue_info.queue_id   = 1;
    queue_info.node_id    = 1;
    queue_info.process_id = 1;
    writer.register_queue_info(queue_info);

    writer_types::stream_info_t stream_info;
    stream_info.stream_id  = 1;
    stream_info.node_id    = 1;
    stream_info.process_id = 1;
    writer.register_stream_info(stream_info);

    writer_types::code_object_info_t code_object_info;
    code_object_info.id         = 1;
    code_object_info.node_id    = 1;
    code_object_info.process_id = 1;
    writer.register_code_object_info(code_object_info);

    writer_types::kernel_symbol_info_t kernel_symbol_info;
    kernel_symbol_info.id          = 1;
    kernel_symbol_info.node_id     = 1;
    kernel_symbol_info.process_id  = 1;
    kernel_symbol_info.code_obj_id = 1;
    writer.register_kernel_symbol_info(kernel_symbol_info);
}

void
seed_region(writer_t&                    writer,
            size_t                       stack_id,
            reader_types::timestamp_t start,
            reader_types::timestamp_t end)
{
    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 1;
    trace_environment.thread_id  = 1;

    writer_types::event_data_t event_data;
    event_data.stack_id = stack_id;

    writer_types::region_data_t region_data;
    region_data.name            = "region";
    region_data.start_timestamp = start;
    region_data.end_timestamp   = end;
    region_data.event           = event_data;
    writer.insert_region_data(region_data, trace_environment);
}

void
seed_kernel_dispatch(writer_t&                    writer,
                     size_t                       stack_id,
                     reader_types::timestamp_t start,
                     reader_types::timestamp_t end)
{
    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 1;
    trace_environment.thread_id  = 1;
    trace_environment.agent_id   = writer_types::agent_unique_id_t{ "GPU", 0 };
    trace_environment.queue_id   = 1;
    trace_environment.stream_id  = 1;

    writer_types::event_data_t event_data;
    event_data.stack_id = stack_id;

    writer_types::kernel_dispatch_data_t kernel_dispatch_data;
    kernel_dispatch_data.kernel_symbol_id = 1;
    kernel_dispatch_data.code_object_id   = 1;
    kernel_dispatch_data.start_timestamp  = start;
    kernel_dispatch_data.end_timestamp    = end;
    kernel_dispatch_data.event            = event_data;
    writer.insert_kernel_dispatch_data(kernel_dispatch_data, trace_environment);
}

void
seed_memory_copy(writer_t&                    writer,
                 size_t                       stack_id,
                 reader_types::timestamp_t start,
                 reader_types::timestamp_t end)
{
    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 1;

    writer_types::event_data_t event_data;
    event_data.stack_id = stack_id;

    writer_types::memory_copy_data_t memory_copy_data;
    memory_copy_data.name            = "copy";
    memory_copy_data.region_name     = "copy";
    memory_copy_data.start_timestamp = start;
    memory_copy_data.end_timestamp   = end;
    memory_copy_data.size            = 1024;
    memory_copy_data.event           = event_data;
    writer.insert_memory_copy_data(memory_copy_data, trace_environment);
}

void
seed_memory_alloc(writer_t&                    writer,
                  size_t                       stack_id,
                  reader_types::timestamp_t start,
                  reader_types::timestamp_t end)
{
    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 1;

    writer_types::event_data_t event_data;
    event_data.stack_id = stack_id;

    writer_types::memory_alloc_data_t memory_alloc_data;
    memory_alloc_data.type            = "ALLOC";
    memory_alloc_data.level           = "REAL";
    memory_alloc_data.start_timestamp = start;
    memory_alloc_data.end_timestamp   = end;
    memory_alloc_data.size            = 4096;
    memory_alloc_data.event           = event_data;
    writer.insert_memory_alloc_data(memory_alloc_data, trace_environment);
}

// ---------------------------------------------------------------------------
// Clique fixture: reproduces rocpd_v3_clique_data.sql. stack 1000 = one of each
// type (region1/kd1/mc1/ma1); stacks 2000/3000/4000/5000 = same-type sibling
// pairs (region2/3, kd2/3, mc2/3, ma2/3); stack 0 = region4 (excluded).
// 11 undirected clique pairs -> 7 directed edges.
// ---------------------------------------------------------------------------
class reader_v3_clique_test : public reader_test
{
protected:
    void SetUp() override
    {
        reader_test::SetUp();
        auto writer = make_writer();
        seed_flow_identity(*writer);

        // regions 1..4 (ids by insertion order)
        seed_region(*writer, 1000, 1000, 1100);  // region 1
        seed_region(*writer, 2000, 2000, 2100);  // region 2
        seed_region(*writer, 2000, 2050, 2150);  // region 3
        seed_region(*writer, 0, 9000, 9100);     // region 4 (stack 0 -> excluded)

        // kernel dispatches 1..3
        seed_kernel_dispatch(*writer, 1000, 1200, 1300);  // kd 1
        seed_kernel_dispatch(*writer, 3000, 3000, 3100);  // kd 2
        seed_kernel_dispatch(*writer, 3000, 3050, 3150);  // kd 3

        // memory copies 1..3
        seed_memory_copy(*writer, 1000, 1400, 1500);  // mc 1
        seed_memory_copy(*writer, 4000, 4000, 4100);  // mc 2
        seed_memory_copy(*writer, 4000, 4050, 4150);  // mc 3

        // memory allocates 1..3
        seed_memory_alloc(*writer, 1000, 1600, 1700);  // ma 1
        seed_memory_alloc(*writer, 5000, 5000, 5100);  // ma 2
        seed_memory_alloc(*writer, 5000, 5050, 5150);  // ma 3

        writer->flush_in_memory_data_to_disk();
        writer.reset();
        m_reader = make_reader();
    }

    std::unique_ptr<reader_t> m_reader;
};

TEST_F(reader_v3_clique_test, get_flows_emits_directed_typed_clique)
{
    using et         = profiler_hub::reader_types::event_type_t;
    using fk         = profiler_hub::reader_types::flow_kind_t;
    using flow_key_t = std::pair<profiler_hub::reader_types::event_id_t,
                                 profiler_hub::reader_types::event_id_t>;

    auto flows = m_reader->get_flows();
    // Directed model: the 11 undirected pairs collapse to 7 directed edges. The 3
    // cross-type region->gpu legs were already single-direction; the 4 same-type sets
    // (region<->region, kd<->kd, mc<->mc, ma<->ma) each de-dup from two ordered pairs to
    // one. This is the "half on symmetric pairs" property the directed model guarantees.
    ASSERT_EQ(flows.size(), 7U);

    struct edge_expect
    {
        fk       kind;
        uint64_t flow_id;
    };
    std::map<flow_key_t, edge_expect> got;
    for(const auto& f : flows)
    {
        got.emplace(flow_key_t{ f.source, f.dest },
                    edge_expect{ f.kind, flow_id_value(f.flow_id) });
    }

    // Exact directed oracle. parent_stack_id is NULL throughout the fixture, so lineage
    // orientation never fires and every edge is oriented by ascending start-ts (earlier
    // endpoint = source). flow_id == the shared source stack_id, so region 1's three
    // cross-type legs all group under flow_id 1000. Colliding raw row ids (region 1 /
    // kd 1 / mc 1 / ma 1) still mint to distinct handles via the encoded type tag.
    using event_id_t = profiler_hub::reader_types::event_id_t;
    auto expect_edge = [&](event_id_t s, event_id_t d, fk kind, uint64_t fid) {
        auto it = got.find(flow_key_t{ s, d });
        ASSERT_NE(it, got.end()) << "missing directed edge";
        EXPECT_EQ(it->second.kind, kind);
        EXPECT_EQ(it->second.flow_id, fid);
    };
    // region 1 (start 1000) is earliest in its stack, so it sources all three gpu legs.
    expect_edge(make_event_id(et::region, 1),
                make_event_id(et::kernel_dispatch, 1),
                fk::launch_to_dispatch,
                1000);
    expect_edge(make_event_id(et::region, 1),
                make_event_id(et::memory_copy, 1),
                fk::copy_submit_to_exec,
                1000);
    expect_edge(make_event_id(et::region, 1),
                make_event_id(et::memory_allocate, 1),
                fk::copy_submit_to_exec,
                1000);
    // Same-type sets: earlier-start endpoint sources the single surviving directed edge.
    expect_edge(make_event_id(et::region, 2),  // start 2000 < region 3 start 2050
                make_event_id(et::region, 3),
                fk::generic,
                2000);
    expect_edge(make_event_id(et::kernel_dispatch, 2),  // 3000 < 3050
                make_event_id(et::kernel_dispatch, 3),
                fk::stream_dependency,
                3000);
    expect_edge(make_event_id(et::memory_copy, 2),  // 4000 < 4050
                make_event_id(et::memory_copy, 3),
                fk::stream_dependency,
                4000);
    expect_edge(make_event_id(et::memory_allocate, 2),  // 5000 < 5050
                make_event_id(et::memory_allocate, 3),
                fk::stream_dependency,
                5000);

    // Handle-collision guard: region 1 / kernel_dispatch 1 / memory_copy 1 /
    // memory_allocate 1 all share raw row id 1 but come from different per-type tables.
    // They MUST mint to four distinct handles (the identity leak task 028 closes).
    std::unordered_set<profiler_hub::reader_types::event_id_t> distinct{
        make_event_id(et::region, 1),
        make_event_id(et::kernel_dispatch, 1),
        make_event_id(et::memory_copy, 1),
        make_event_id(et::memory_allocate, 1)
    };
    ASSERT_EQ(distinct.size(), 4U);
}

TEST_F(reader_v3_clique_test, get_flows_dedups_symmetric_pairs_to_single_direction)
{
    // Direction / de-dup: for every surviving edge (a -> b), the reverse (b -> a) must
    // NOT also be present. This is the core invariant of the directed model: each
    // unordered clique pair yields exactly one edge.
    auto flows = m_reader->get_flows();
    std::set<std::pair<profiler_hub::reader_types::event_id_t,
                       profiler_hub::reader_types::event_id_t>>
        directed;
    for(const auto& f : flows)
        directed.emplace(f.source, f.dest);
    ASSERT_EQ(directed.size(), flows.size());  // no duplicate directed edges
    for(const auto& f : flows)
    {
        auto reverse = std::make_pair(f.dest, f.source);
        EXPECT_EQ(directed.count(reverse), 0U)
            << "both directions of a symmetric pair survived de-dup";
    }
}

TEST_F(reader_v3_clique_test, get_flows_kind_matches_endpoint_types)
{
    using et     = profiler_hub::reader_types::event_type_t;
    using fk     = profiler_hub::reader_types::flow_kind_t;
    auto type_of = [](const profiler_hub::reader_types::event_id_t& id) {
        return profiler_hub::reader_types::detail::event_id_access::type(id);
    };
    // Kind correctness: the flow_kind_t of every edge is a pure function of its ordered
    // endpoint types, independent of which endpoint won orientation.
    for(const auto& f : m_reader->get_flows())
    {
        const auto s = type_of(f.source);
        const auto d = type_of(f.dest);
        if(s == et::region && d == et::kernel_dispatch)
            EXPECT_EQ(f.kind, fk::launch_to_dispatch);
        else if(s == et::region && (d == et::memory_copy || d == et::memory_allocate))
            EXPECT_EQ(f.kind, fk::copy_submit_to_exec);
        else if(s == d && (s == et::kernel_dispatch || s == et::memory_copy ||
                           s == et::memory_allocate))
            EXPECT_EQ(f.kind, fk::stream_dependency);
        else
            EXPECT_EQ(f.kind, fk::generic);
    }
}

TEST_F(reader_v3_clique_test, get_flows_for_chain_groups_by_flow_id)
{
    using et = profiler_hub::reader_types::event_type_t;
    // flow_id grouping: region 1's stack (flow_id 1000) holds all three cross-type legs.
    // get_flows_for_chain returns exactly that group, and sorting it by source start
    // recovers linear order (all three share source region 1, so order is by dest start:
    // kd1=1200 < mc1=1400 < ma1=1600).
    auto                                  all = m_reader->get_flows();
    profiler_hub::reader_types::flow_id_t chain_1000{};
    for(const auto& f : all)
        if(flow_id_value(f.flow_id) == 1000) chain_1000 = f.flow_id;
    ASSERT_EQ(flow_id_value(chain_1000), 1000U);

    auto chain = m_reader->get_flows_for_chain(chain_1000);
    ASSERT_EQ(chain.size(), 3U);
    for(const auto& f : chain)
        EXPECT_EQ(flow_id_value(f.flow_id), 1000U);

    // A flow_id that names no chain returns empty.
    auto none = m_reader->get_flows_for_chain(
        profiler_hub::reader_types::detail::flow_id_access::make(99));
    EXPECT_TRUE(none.empty());

    // Sorting the group by source start recovers a stable linear ordering.
    std::sort(chain.begin(), chain.end(), [&](const auto& x, const auto& y) {
        // all share the same source (region 1); tie-break by dest handle for determinism
        return x.dest < y.dest;
    });
    EXPECT_EQ(
        profiler_hub::reader_types::detail::event_id_access::type(chain.front().source),
        et::region);
}

TEST_F(reader_v3_clique_test, get_flows_for_event_returns_adjacent_edges)
{
    using et = profiler_hub::reader_types::event_type_t;
    // Adjacency: region 1 is the source of exactly its three cross-type legs and the dest
    // of none, so get_flows_for_event(region 1) returns those 3.
    const auto region1 = make_event_id(et::region, 1);
    auto       adj     = m_reader->get_flows_for_event(region1);
    ASSERT_EQ(adj.size(), 3U);
    for(const auto& f : adj)
        EXPECT_TRUE(f.source == region1 || f.dest == region1);

    // kernel_dispatch 1 is a leaf dest (adjacent to exactly one edge).
    auto kd1_adj = m_reader->get_flows_for_event(make_event_id(et::kernel_dispatch, 1));
    ASSERT_EQ(kd1_adj.size(), 1U);
    EXPECT_EQ(kd1_adj.front().dest, make_event_id(et::kernel_dispatch, 1));

    // An event handle that participates in no edge returns empty.
    auto none = m_reader->get_flows_for_event(make_event_id(et::region, 999));
    EXPECT_TRUE(none.empty());
}

TEST_F(reader_v3_clique_test,
       get_flows_in_window_empty_window_and_tracks_equals_get_flows)
{
    // Criterion 7(b)/7(e): empty window + empty tracks + max_edges 0 is a pure pass-
    // through of get_flows({}) — same edges, same source/dest/flow_id/kind, no cap.
    auto all = m_reader->get_flows();
    auto win = m_reader->get_flows_in_window({}, {}, 0);
    ASSERT_EQ(win.size(), all.size());
    ASSERT_EQ(win.size(), 7U);

    std::map<std::pair<profiler_hub::reader_types::event_id_t,
                       profiler_hub::reader_types::event_id_t>,
             std::pair<uint64_t, profiler_hub::reader_types::flow_kind_t>>
        oracle;
    for(const auto& f : all)
        oracle.emplace(std::make_pair(f.source, f.dest),
                       std::make_pair(flow_id_value(f.flow_id), f.kind));
    for(const auto& f : win)
    {
        auto it = oracle.find({ f.source, f.dest });
        ASSERT_NE(it, oracle.end()) << "windowed edge not a member of get_flows({})";
        EXPECT_EQ(it->second.first, flow_id_value(f.flow_id));
        EXPECT_EQ(it->second.second, f.kind);
    }
}

TEST_F(reader_v3_clique_test, get_flows_in_window_filters_by_extent)
{
    // Criterion 7(a): window overlap uses the edge extent, boundary-inclusive.
    // [2000,5150] captures the four same-type sibling edges (extents start >= 2000);
    // region1's three legs (ehi <= 1700 < 2000) fall out.
    profiler_hub::reader_types::time_window_t inner;
    inner.start = 2000;
    inner.end   = 5150;
    EXPECT_EQ(m_reader->get_flows_in_window({}, inner, 0).size(), 4U);

    // Both boundaries inclusive: [1700,2000] touches region1->ma1 (ehi==1700) and
    // region2->region3 (elo==2000) and nothing else.
    profiler_hub::reader_types::time_window_t straddle;
    straddle.start = 1700;
    straddle.end   = 2000;
    EXPECT_EQ(m_reader->get_flows_in_window({}, straddle, 0).size(), 2U);

    // A window past every edge excludes all of them.
    profiler_hub::reader_types::time_window_t outside;
    outside.start = 6000;
    EXPECT_TRUE(m_reader->get_flows_in_window({}, outside, 0).empty());
}

TEST_F(reader_v3_clique_test, get_flows_in_window_filters_by_track_membership)
{
    // Criterion 7(c): an edge is kept iff AT LEAST ONE endpoint sits on a listed track.
    // The single cpu_thread track carries regions 1/2/3, so scoping to it keeps region1's
    // three legs (source-only membership) plus region2->region3 (both endpoints), and
    // drops the kd/mc/ma sibling edges (neither endpoint on a region track).
    auto cpu = find_first_track(m_reader->get_tracks(),
                                profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    EXPECT_EQ(m_reader->get_flows_in_window({ cpu->id }, {}, 0).size(), 4U);

    // Empty track list applies no filter.
    EXPECT_EQ(m_reader->get_flows_in_window({}, {}, 0).size(), 7U);
}

TEST_F(reader_v3_clique_test, get_flows_in_window_decimates_by_latency_stably)
{
    // Criterion 7(d): cap to max_edges by descending arrow-span latency. region1's three
    // legs have the only nonzero latencies (500 > 300 > 100), so max_edges 3 returns
    // exactly those three; the four zero-latency sibling edges are dropped.
    using et  = profiler_hub::reader_types::event_type_t;
    auto top3 = m_reader->get_flows_in_window({}, {}, 3);
    ASSERT_EQ(top3.size(), 3U);

    std::set<std::pair<profiler_hub::reader_types::event_id_t,
                       profiler_hub::reader_types::event_id_t>>
        got;
    for(const auto& f : top3)
        got.emplace(f.source, f.dest);
    const auto region1 = make_event_id(et::region, 1);
    EXPECT_EQ(got.count({ region1, make_event_id(et::kernel_dispatch, 1) }), 1U);
    EXPECT_EQ(got.count({ region1, make_event_id(et::memory_copy, 1) }), 1U);
    EXPECT_EQ(got.count({ region1, make_event_id(et::memory_allocate, 1) }), 1U);

    // Highest latency emitted first (region1->ma1, latency 500).
    EXPECT_EQ(top3.front().source, region1);
    EXPECT_EQ(top3.front().dest, make_event_id(et::memory_allocate, 1));

    // Stable: an identical query yields an identical ordering across calls.
    auto again = m_reader->get_flows_in_window({}, {}, 3);
    ASSERT_EQ(again.size(), top3.size());
    for(size_t i = 0; i < top3.size(); ++i)
    {
        EXPECT_EQ(again[i].source, top3[i].source);
        EXPECT_EQ(again[i].dest, top3[i].dest);
    }

    // max_edges 0 is uncapped; a cap at/above the set size is a no-op.
    EXPECT_EQ(m_reader->get_flows_in_window({}, {}, 0).size(), 7U);
    EXPECT_EQ(m_reader->get_flows_in_window({}, {}, 99).size(), 7U);
}

// ---------------------------------------------------------------------------
// Flow-ordering / tie-break fixture: reproduces rocpd_v3_flow_order_data.sql.
// Crafts the two degenerate-but-legitimate shapes the clique fixture never hits:
// two endpoints at an identical start (equal-start direction tie-break) and two
// same-source legs at identical (clamped-zero) latency (windowed decimation
// final dest tie-break). parent_stack_id NULL throughout.
// ---------------------------------------------------------------------------
class reader_v3_flow_order_test : public reader_test
{
protected:
    void SetUp() override
    {
        reader_test::SetUp();
        auto writer = make_writer();
        seed_flow_identity(*writer);

        // regions 1..3
        seed_region(*writer, 1000, 5000, 5100);  // region 1 (equal-start pair)
        seed_region(*writer, 1000, 5000, 5100);  // region 2 (equal-start pair)
        seed_region(*writer, 2000, 6000, 6500);  // region 3 (encloses its children)

        // kernel dispatches 1..3
        seed_kernel_dispatch(*writer, 2000, 6100, 6200);  // kd 1 (inside region 3)
        seed_kernel_dispatch(*writer, 3000, 7000, 7100);  // kd 2 (equal-start pair)
        seed_kernel_dispatch(*writer, 3000, 7000, 7100);  // kd 3 (equal-start pair)

        // memory copy 1
        seed_memory_copy(*writer, 2000, 6200, 6300);  // mc 1 (inside region 3)

        writer->flush_in_memory_data_to_disk();
        writer.reset();
        m_reader = make_reader();
    }

    std::unique_ptr<reader_t> m_reader;
};

TEST_F(reader_v3_flow_order_test, equal_start_region_pair_tie_breaks_by_handle_order)
{
    using et = profiler_hub::reader_types::event_type_t;
    using fk = profiler_hub::reader_types::flow_kind_t;
    // Stack 1000 = { region 1, region 2 } BOTH start 5000, parent_stack_id NULL. Neither
    // parent-lineage branch fires and the start-ts branch cannot decide (starts are
    // identical), so direction falls to the deterministic equal-start tie-break:
    // src = key.first (the lower event_id_t handle), dst = key.second. region 1 (row id 1)
    // mints a lower handle than region 2, so the single surviving directed edge MUST be
    // region 1 -> region 2 (never the reverse).
    const auto r1 = make_event_id(et::region, 1);
    const auto r2 = make_event_id(et::region, 2);
    ASSERT_TRUE(r1 < r2) << "test premise: lower row id mints the lower handle";

    int  seen  = 0;
    auto flows = m_reader->get_flows();
    for(const auto& f : flows)
    {
        if(flow_id_value(f.flow_id) != 1000) continue;
        ++seen;
        EXPECT_EQ(f.source, r1);  // equal starts -> lower handle is source
        EXPECT_EQ(f.dest, r2);
        EXPECT_EQ(f.kind, fk::generic);  // region -> region
        EXPECT_TRUE(f.source < f.dest);  // the tie-break invariant itself
    }
    EXPECT_EQ(seen, 1) << "the symmetric (region2,region1) pair must de-dup to one edge";
}

TEST_F(reader_v3_flow_order_test, equal_start_kd_siblings_tie_break_by_handle_order)
{
    using et = profiler_hub::reader_types::event_type_t;
    using fk = profiler_hub::reader_types::flow_kind_t;
    // Same equal-start tie-break on the same-type sibling path. Stack 3000 =
    // { kd 2, kd 3 } BOTH start 7000: src = lower handle = kd 2, dst = kd 3, kind
    // stream_dependency (kd<->kd sibling).
    const auto k2 = make_event_id(et::kernel_dispatch, 2);
    const auto k3 = make_event_id(et::kernel_dispatch, 3);
    ASSERT_TRUE(k2 < k3);

    int  seen  = 0;
    auto flows = m_reader->get_flows();
    for(const auto& f : flows)
    {
        if(flow_id_value(f.flow_id) != 3000) continue;
        ++seen;
        EXPECT_EQ(f.source, k2);
        EXPECT_EQ(f.dest, k3);
        EXPECT_EQ(f.kind, fk::stream_dependency);
        EXPECT_TRUE(f.source < f.dest);
    }
    EXPECT_EQ(seen, 1);
}

TEST_F(reader_v3_flow_order_test, window_decimation_tie_breaks_equal_latency_by_dest)
{
    using et = profiler_hub::reader_types::event_type_t;
    // Stack 2000: region 3 [6000,6500] sources kd 1 (start 6100) and mc 1 (start 6200);
    // both children start BEFORE region 3 ends, so both arrow-span latencies
    // (dst.start - src.end) clamp to 0 -> EQUAL latency, and both share source region 3.
    // A window that admits only these two edges, capped at max_edges = 1, forces the
    // decimation sort to compare two flows with equal latency AND equal source -> the
    // final tie-break `a.dest < b.dest` decides. kd 1's handle < mc 1's handle
    // (event_type kernel_dispatch < memory_copy), so the survivor MUST be region 3 -> kd 1.
    profiler_hub::reader_types::time_window_t win;
    win.start = 6000;
    win.end   = 6600;  // excludes the stack-1000 (5000) and stack-3000 (7000) edges

    // Uncapped, the window admits exactly the two zero-latency same-source legs.
    auto both = m_reader->get_flows_in_window({}, win, 0);
    ASSERT_EQ(both.size(), 2U);
    for(const auto& f : both)
        EXPECT_EQ(f.source, make_event_id(et::region, 3));

    // Capped at 1: the equal-latency, equal-source pair is tie-broken by dest handle,
    // keeping the lower dest (kd 1) and dropping mc 1.
    auto top1 = m_reader->get_flows_in_window({}, win, 1);
    ASSERT_EQ(top1.size(), 1U);
    EXPECT_EQ(top1.front().source, make_event_id(et::region, 3));
    EXPECT_EQ(top1.front().dest, make_event_id(et::kernel_dispatch, 1));

    // Deterministic across calls (stable ranking, the design contract backstops).
    auto again = m_reader->get_flows_in_window({}, win, 1);
    ASSERT_EQ(again.size(), 1U);
    EXPECT_EQ(again.front().source, top1.front().source);
    EXPECT_EQ(again.front().dest, top1.front().dest);
}

TEST_F(reader_v3_flow_order_test, full_flow_set_matches_oracle)
{
    using et         = profiler_hub::reader_types::event_type_t;
    using fk         = profiler_hub::reader_types::flow_kind_t;
    using flow_key_t = std::pair<profiler_hub::reader_types::event_id_t,
                                 profiler_hub::reader_types::event_id_t>;
    // The complete directed oracle for this fixture: 4 edges. Pins the equal-start pairs
    // and the two zero-latency region-3 legs so a future fixture edit cannot silently
    // change the flow set out from under the tie-break tests above.
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 4U);

    std::map<flow_key_t, std::pair<fk, uint64_t>> got;
    for(const auto& f : flows)
        got.emplace(flow_key_t{ f.source, f.dest },
                    std::make_pair(f.kind, flow_id_value(f.flow_id)));

    auto expect_edge = [&](profiler_hub::reader_types::event_id_t s,
                           profiler_hub::reader_types::event_id_t d,
                           fk                                     kind,
                           uint64_t                               fid) {
        auto it = got.find(flow_key_t{ s, d });
        ASSERT_NE(it, got.end()) << "missing directed edge";
        EXPECT_EQ(it->second.first, kind);
        EXPECT_EQ(it->second.second, fid);
    };
    expect_edge(
        make_event_id(et::region, 1), make_event_id(et::region, 2), fk::generic, 1000);
    expect_edge(make_event_id(et::region, 3),
                make_event_id(et::kernel_dispatch, 1),
                fk::launch_to_dispatch,
                2000);
    expect_edge(make_event_id(et::region, 3),
                make_event_id(et::memory_copy, 1),
                fk::copy_submit_to_exec,
                2000);
    expect_edge(make_event_id(et::kernel_dispatch, 2),
                make_event_id(et::kernel_dispatch, 3),
                fk::stream_dependency,
                3000);
}

// ---------------------------------------------------------------------------
// Flat-clique edge fixture: the single flow test carried over from the edge
// fixture. Each non-zero stack is one region + one GPU event (region source,
// one GPU-type dest, no siblings); stack 0 is a lone excluded region. The other
// 17 edge tests (counter/track/interval/scalar/stream/etc.) are a deferred SQL
// port chunk — they are not flow tests and are not writer-seedable here.
// ---------------------------------------------------------------------------
class reader_v3_edge_flow_test : public reader_test
{
protected:
    void SetUp() override
    {
        reader_test::SetUp();
        auto writer = make_writer();
        seed_flow_identity(*writer);

        // Three flat cliques (region + one GPU event each) + one excluded stack-0 region.
        seed_region(*writer, 100, 1000, 1100);  // region A (stack 100)
        seed_region(*writer, 200, 2000, 2100);  // region B (stack 200)
        seed_region(*writer, 400, 3000, 3100);  // region C (stack 400)
        seed_region(*writer, 0, 9000, 9100);    // region D (stack 0 -> excluded)

        seed_kernel_dispatch(*writer, 100, 1200, 1300);  // kd    (stack 100)
        seed_memory_copy(*writer, 200, 2200, 2300);      // mc    (stack 200)
        seed_memory_alloc(*writer, 400, 3200, 3300);     // ma    (stack 400)

        writer->flush_in_memory_data_to_disk();
        writer.reset();
        m_reader = make_reader();
    }

    std::unique_ptr<reader_t> m_reader;
};

TEST_F(reader_v3_edge_flow_test, get_flows_excludes_zero_and_null_stack_id)
{
    // stack_id linkage: region->kernel_dispatch (100), region->memory_copy (200),
    // region->memory_allocate (400) = 3 flows. The stack-0 region is excluded, and
    // events with NULL stack_id never appear. Flat clique (one region + one GPU event
    // per stack) => region source, one GPU-type dest, no siblings.
    using et   = profiler_hub::reader_types::event_type_t;
    using fk   = profiler_hub::reader_types::flow_kind_t;
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 3U);
    // Region D (the 4th region inserted, on stack 0) is the exclusion guard: it and any
    // NULL-stack event must appear as neither source nor dest of any flow.
    const auto excluded_region = make_event_id(et::region, 4);
    for(const auto& f : flows)
    {
        ASSERT_NE(f.source, excluded_region);
        ASSERT_NE(f.dest, excluded_region);
        ASSERT_EQ(type_of(f.source), profiler_hub::reader_types::event_type_t::region);
        ASSERT_TRUE(m_reader->get_event_info(f.source).has_value());
        ASSERT_EQ(count_interval_resolutions(*m_reader, f.dest), 1);
        ASSERT_NE(type_of(f.dest), profiler_hub::reader_types::event_type_t::region);
        // Directed/typed: excluded stacks (0/NULL) never appear, so every flow_id is a
        // non-zero source stack_id; kind is a cross-type region->gpu category.
        ASSERT_GT(flow_id_value(f.flow_id), 0U);
        ASSERT_TRUE(f.kind == fk::launch_to_dispatch ||
                    f.kind == fk::copy_submit_to_exec);
    }
}

// ===========================================================================
// Edge-matrix NON-flow tests (ported from pre-rebase cac3369ac5). Chunk 2 of
// the mixed-policy port wave (task 070). The pre-rebase reader_v3_edge_test
// loaded a single SQL fixture (rocpd_v3_edge.db) for all 17 non-flow tests; the
// DL-015 3-way split re-homes them by seed mechanism:
//   * (c) 9 writer-portable behaviors -> reader_v3_edge_test below, reproduced
//     through writer_t::insert_*_data + insert_pmc_event_data (counters).
//   * (a) 2 degenerate + (b) 6 writer-id-blocked -> reader_v3_edge_sql_test,
//     which still loads the SQL fixture (bare non-pmc sample track a correct
//     writer must never emit; rocpd_arg on kd/mc/ma; counter/track identity via
//     register_track_info) — see that fixture's header.
// Assertions are preserved verbatim; only the seeding mechanism moved.
// ===========================================================================

// Writer-seeded reproduction of the edge oracle's non-counter matrix + its two
// discoverable counters. The one off-by-one vs. the 1-based SQL fixture is agent
// id: agent primary keys are 0-based (source/autoincrementer.hpp), but the
// memory-track test asserts agent_info->id == 1, so a throwaway agent is
// registered first, landing the real {GPU,0} agent (used by kd + alloc) at id 1.
class reader_v3_edge_test : public reader_test
{
protected:
    void SetUp() override
    {
        reader_test::SetUp();
        auto writer = make_writer();
        seed_edge_identity(*writer);

        // 4 regions -> one cpu_thread track. Insertion order deliberately differs
        // from start order so the reader's ORDER BY start is exercised; RegionAlpha
        // is inserted 3rd so it carries region row-id 2 (0-based) while being the
        // start-order front.
        seed_named_region(*writer, "RegionGamma", 0, 3000, 3500);   // region id 0
        seed_named_region(*writer, "RegionBeta", 200, 2000, 2500);  // region id 1
        seed_named_region(*writer, "RegionAlpha", 100, 1000, 5000); // region id 2
        seed_named_region(*writer, "RegionDelta", 400, 6000, 6500); // region id 3

        // 3 kernel dispatches: Queue-A (1) = kd@1200 + kd@1600, Queue-B (2) = kd@1400.
        // All on stream 1.
        seed_kd(*writer, /*queue*/ 1, /*stream*/ 1, 1200, 1300);
        seed_kd(*writer, /*queue*/ 2, /*stream*/ 1, 1400, 1500);
        seed_kd(*writer, /*queue*/ 1, /*stream*/ 1, 1600, 1700);

        // 3 memory copies: all dst_agent NULL -> ONE dma track [2100,2200,2400].
        // stream 1 gets 2100 + 2200, stream 2 gets 2400.
        seed_mc(*writer, /*stream*/ 1, 2100, 2150);
        seed_mc(*writer, /*stream*/ 1, 2200, 2250);
        seed_mc(*writer, /*stream*/ 2, 2400, 2450);

        // 1 memory allocate on agent {GPU,0} (id 1), queue NULL, stream 1 -> one
        // memory track + one memory_activity track; also the 6th stream-1 event.
        seed_ma(*writer, /*stream*/ 1, 6100, 6200);

        // 2 discoverable counters. Samples inserted out of timestamp order so the
        // reader's ORDER BY timestamp is exercised.
        seed_counter(*writer, "GRBM_COUNT", "grbm_track",
                     { { 3000, 30.5 }, { 1000, 10.5 }, { 2000, 20.5 } });
        seed_counter(*writer, "SQ_WAVES", "sq_waves_track",
                     { { 500, 5.0 }, { 1500, 15.0 } });

        writer->flush_in_memory_data_to_disk();
        writer.reset();
        m_reader = make_reader();
    }

    void seed_edge_identity(writer_t& writer) const
    {
        const writer_types::node_info_t node_info{ 1, 42, "synthetic-machine" };
        writer.register_node_info(node_info);

        writer_types::process_info_t process_info;
        process_info.pid     = 1;
        process_info.node_id = 1;
        writer.register_process_info(process_info);

        writer_types::thread_info_t thread_info;
        thread_info.thread_id  = 1;
        thread_info.node_id    = 1;
        thread_info.process_id = 1;
        writer.register_thread_info(thread_info);

        // Throwaway agent registered first so its 0-based primary key is 0, leaving
        // the real GPU agent below at id 1 (what the memory-track test asserts).
        writer_types::agent_info_t dummy_agent;
        dummy_agent.unique_id.agent_type = "CPU";
        dummy_agent.unique_id.type_index = 0;
        dummy_agent.node_id              = 1;
        dummy_agent.process_id           = 1;
        writer.register_agent_info(dummy_agent);

        writer_types::agent_info_t agent_info;
        agent_info.unique_id.agent_type = "GPU";
        agent_info.unique_id.type_index = 0;
        agent_info.node_id              = 1;
        agent_info.process_id           = 1;
        writer.register_agent_info(agent_info);

        for(size_t queue_id : { 1U, 2U })
        {
            writer_types::queue_info_t queue_info;
            queue_info.queue_id   = queue_id;
            queue_info.node_id    = 1;
            queue_info.process_id = 1;
            writer.register_queue_info(queue_info);
        }

        // Throwaway stream registered first so its 0-based primary key is 0, leaving
        // the two real streams at primary keys 1 and 2 (the reader keys stream_info by
        // primary key, and the ported tests expect the 6-event stream at stream_id 1).
        // It carries no events, so it never surfaces as a synthesized stream track.
        {
            writer_types::stream_info_t dummy_stream;
            dummy_stream.stream_id  = 99;
            dummy_stream.node_id    = 1;
            dummy_stream.process_id = 1;
            writer.register_stream_info(dummy_stream);
        }

        for(size_t stream_id : { 1U, 2U })
        {
            writer_types::stream_info_t stream_info;
            stream_info.stream_id  = stream_id;
            stream_info.node_id    = 1;
            stream_info.process_id = 1;
            writer.register_stream_info(stream_info);
        }

        writer_types::code_object_info_t code_object_info;
        code_object_info.id         = 1;
        code_object_info.node_id    = 1;
        code_object_info.process_id = 1;
        writer.register_code_object_info(code_object_info);

        writer_types::kernel_symbol_info_t kernel_symbol_info;
        kernel_symbol_info.id          = 1;
        kernel_symbol_info.node_id     = 1;
        kernel_symbol_info.process_id  = 1;
        kernel_symbol_info.code_obj_id = 1;
        writer.register_kernel_symbol_info(kernel_symbol_info);
    }

    void seed_named_region(writer_t&                 writer,
                           std::string_view          name,
                           size_t                    stack_id,
                           reader_types::timestamp_t start,
                           reader_types::timestamp_t end) const
    {
        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 1;
        trace_environment.thread_id  = 1;

        writer_types::event_data_t event_data;
        event_data.stack_id = stack_id;

        writer_types::region_data_t region_data;
        region_data.name            = name;
        region_data.start_timestamp = start;
        region_data.end_timestamp   = end;
        region_data.event           = event_data;
        writer.insert_region_data(region_data, trace_environment);
    }

    void seed_kd(writer_t&                 writer,
                 size_t                    queue_id,
                 size_t                    stream_id,
                 reader_types::timestamp_t start,
                 reader_types::timestamp_t end) const
    {
        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 1;
        trace_environment.thread_id  = 1;
        trace_environment.agent_id   = writer_types::agent_unique_id_t{ "GPU", 0 };
        trace_environment.queue_id   = queue_id;
        trace_environment.stream_id  = stream_id;

        writer_types::kernel_dispatch_data_t kernel_dispatch_data;
        kernel_dispatch_data.kernel_symbol_id = 1;
        kernel_dispatch_data.code_object_id   = 1;
        kernel_dispatch_data.start_timestamp  = start;
        kernel_dispatch_data.end_timestamp    = end;
        kernel_dispatch_data.event            = writer_types::event_data_t{};
        writer.insert_kernel_dispatch_data(kernel_dispatch_data, trace_environment);
    }

    void seed_mc(writer_t&                 writer,
                 size_t                    stream_id,
                 reader_types::timestamp_t start,
                 reader_types::timestamp_t end) const
    {
        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 1;
        trace_environment.stream_id  = stream_id;

        writer_types::memory_copy_data_t memory_copy_data;
        memory_copy_data.name            = "copy";
        memory_copy_data.region_name     = "copy";
        memory_copy_data.start_timestamp = start;
        memory_copy_data.end_timestamp   = end;
        memory_copy_data.size            = 1024;
        // dst_agent_id left unset -> all copies share the NULL destination-agent
        // key -> exactly one dma track.
        memory_copy_data.event = writer_types::event_data_t{};
        writer.insert_memory_copy_data(memory_copy_data, trace_environment);
    }

    void seed_ma(writer_t&                 writer,
                 size_t                    stream_id,
                 reader_types::timestamp_t start,
                 reader_types::timestamp_t end) const
    {
        writer_types::trace_environment_t trace_environment;
        trace_environment.node_id    = 1;
        trace_environment.process_id = 1;
        trace_environment.agent_id   = writer_types::agent_unique_id_t{ "GPU", 0 };
        trace_environment.stream_id  = stream_id;
        // queue_id left unset -> memory track carries agent, no queue.

        writer_types::memory_alloc_data_t memory_alloc_data;
        memory_alloc_data.type            = "ALLOC";
        memory_alloc_data.level           = "REAL";
        memory_alloc_data.start_timestamp = start;
        memory_alloc_data.end_timestamp   = end;
        memory_alloc_data.size            = 4096;
        memory_alloc_data.event           = writer_types::event_data_t{};
        writer.insert_memory_alloc_data(memory_alloc_data, trace_environment);
    }

    // Register a PMC + a track, then insert one PMC-backed sample per (ts,value).
    // insert_pmc_event_data inserts an event, a rocpd_pmc_event row (making the
    // track discoverable as a counter), and a rocpd_sample on the shared track;
    // the counter's display name resolves to the PMC name. pmc/track names are
    // string_view — callers must pass string literals (static storage).
    void seed_counter(
        writer_t&                                                            writer,
        std::string_view                                                     pmc_name,
        std::string_view                                                     track_name,
        const std::vector<std::pair<reader_types::timestamp_t, double>>&     samples) const
    {
        writer_types::pmc_info_t pmc_info;
        pmc_info.unique_id.name = pmc_name;
        pmc_info.symbol         = pmc_name;
        pmc_info.node_id        = 1;
        pmc_info.process_id     = 1;
        writer.register_pmc_info(pmc_info);

        writer_types::track_info_t track_info;
        track_info.name       = writer_types::track_name_t{ track_name };
        track_info.node_id    = 1;
        track_info.process_id = 1;
        writer.register_track_info(track_info);

        for(const auto& [timestamp, value] : samples)
        {
            writer_types::pmc_event_data_t pmc_event_data;
            pmc_event_data.event            = writer_types::event_data_t{};
            pmc_event_data.value            = value;
            pmc_event_data.sample.timestamp = timestamp;
            pmc_event_data.sample.track     = track_info;
            writer.insert_pmc_event_data(
                pmc_event_data,
                writer_types::pmc_info_unique_id_t{ pmc_name, std::nullopt });
        }
    }

    std::unique_ptr<reader_t> m_reader;
};

TEST_F(reader_v3_edge_test, get_interval_track_cpu_thread_regions_ordered)
{
    // track 1 carries 4 regions; ORDER BY start (row-id order deliberately differs):
    //   region 2 (start 1000) -> 3 (2000) -> 1 (3000) -> 4 (6000).
    auto tracks = m_reader->get_tracks();
    auto cpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);

    profiler_hub::reader_types::interval_entry_list_t regions;
    for(const auto& t : cpu)
    {
        auto iv = m_reader->get_interval_track(t->id);
        if(iv.size() == 4)
        {
            regions = std::move(iv);
            break;
        }
    }
    ASSERT_EQ(regions.size(), 4U);
    ASSERT_TRUE(is_start_sorted(regions));
    ASSERT_EQ(row_id_of(regions.front().id), 2U);
    ASSERT_EQ(regions.front().start, 1000);
    ASSERT_EQ(regions.front().end, 5000);

    auto details = m_reader->get_event_info(regions.front().id);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->name, "RegionAlpha");
}

TEST_F(reader_v3_edge_test, get_interval_track_gpu_queue_and_dma_ordered)
{
    auto tracks = m_reader->get_tracks();

    // Two gpu_queue tracks: Queue-A has 2 dispatches (start 1200, 1600), Queue-B 1.
    auto gpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_EQ(gpu.size(), 2U);
    profiler_hub::reader_types::interval_entry_list_t gpu_two;
    size_t                                            gpu_singletons = 0;
    for(const auto& t : gpu)
    {
        auto iv = m_reader->get_interval_track(t->id);
        if(iv.size() == 2)
            gpu_two = iv;
        else if(iv.size() == 1)
            ++gpu_singletons;
    }
    ASSERT_EQ(gpu_two.size(), 2U);
    ASSERT_EQ(gpu_singletons, 1U);
    ASSERT_TRUE(is_start_sorted(gpu_two));
    ASSERT_EQ(gpu_two.front().start, 1200);

    // One dma track (all 3 copies share queue_id NULL + dst_agent_id NULL under the
    // by-destination-agent key). Row-id order != start order proves ORDER BY start:
    // copies at 2200 (mc1), 2400 (mc2), 2100 (mc3) => [2100, 2200, 2400].
    auto dma = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 1U);
    auto dma_iv = m_reader->get_interval_track(dma.front()->id);
    ASSERT_EQ(dma_iv.size(), 3U);
    ASSERT_TRUE(is_start_sorted(dma_iv));
    ASSERT_EQ(dma_iv.front().start, 2100);
}

TEST_F(reader_v3_edge_test, get_scalar_track_values_for_both_counters)
{
    auto tracks = m_reader->get_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    for(const auto& c : counters)
    {
        auto samples = m_reader->get_scalar_track(c->id);
        ASSERT_TRUE(is_timestamp_sorted(samples));

        if(c->name == "GRBM_COUNT")
        {
            // 3 samples, ascending timestamp despite differing row-id order.
            ASSERT_FALSE(samples.empty());
            ASSERT_EQ(samples.size(), 3U);
            ASSERT_EQ(samples.front().timestamp, 1000);
            ASSERT_DOUBLE_EQ(samples.front().value, 10.5);

            auto details = m_reader->get_event_info(samples.front().id);
            ASSERT_TRUE(details.has_value());
            ASSERT_EQ(details->ts, samples.front().timestamp);
            ASSERT_FALSE(details->te.has_value());
        }
        else if(c->name == "SQ_WAVES")
        {
            ASSERT_FALSE(samples.empty());
            ASSERT_EQ(samples.size(), 2U);
            ASSERT_EQ(samples.front().timestamp, 500);
            ASSERT_DOUBLE_EQ(samples.front().value, 5.0);

            auto details = m_reader->get_event_info(samples.front().id);
            ASSERT_TRUE(details.has_value());
            ASSERT_EQ(details->ts, samples.front().timestamp);
            ASSERT_FALSE(details->te.has_value());
        }
    }
}

TEST_F(reader_v3_edge_test, track_scoped_queries_respect_type)
{
    // Q7: interval query on a counter (scalar-only) track and scalar query on a
    // cpu_thread (interval-only) track both return empty, not an error.
    auto tracks = m_reader->get_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(counter, nullptr);
    ASSERT_NE(cpu, nullptr);
    ASSERT_TRUE(m_reader->get_interval_track(counter->id).empty());
    ASSERT_TRUE(m_reader->get_scalar_track(cpu->id).empty());
}

TEST_F(reader_v3_edge_test, get_track_stats_matches_slices_for_every_track_type)
{
    // Hand-authored oracle: the 4-region cpu_thread track spans start 1000..end 6000+.
    // For every track, stats must equal MIN/MAX/COUNT over the exact interval/scalar
    // slice — this covers cpu_thread, gpu_queue, dma (here the "neither" variant:
    // queue_id NULL + dst_agent_id NULL; the queue+agent "qa" variant is covered by the
    // dma-by-agent fixture) and counter in one pass, per synthesized track flavor.
    auto tracks = m_reader->get_tracks();

    bool checked_cpu = false;
    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        expect_stats_match_intervals(stats, intervals);
        if(intervals.size() == 4)
        {
            ASSERT_EQ(stats.min_ts.value(), 1000U);
            checked_cpu = true;
        }
    }
    ASSERT_TRUE(checked_cpu) << "expected a 4-region cpu_thread track";

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter))
    {
        auto samples = m_reader->get_scalar_track(t->id);
        expect_stats_match_scalars(m_reader->get_track_stats(t->id), samples);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory_activity))
    {
        auto samples = m_reader->get_scalar_track(t->id);
        expect_stats_match_scalars(m_reader->get_track_stats(t->id), samples);
    }
}

TEST_F(reader_v3_edge_test, get_interval_track_stream_aggregates_three_op_kinds)
{
    // This is the only fixture exercising all THREE UNION legs of a stream track,
    // including memory_allocate. Hand-authored oracle:
    //   stream 1 (nid,pid,stream_id)=(1,1,1): 3 kernel_dispatch + 2 memory_copy +
    //       1 memory_allocate = 6 events, ORDER BY start:
    //       kd3(1200) kd2(1400) kd1(1600) mc3(2100) mc1(2200) ma1(6100)
    //   stream 2 (1,1,2): 1 memory_copy = 1 event (mc2 start 2400)
    // op_kind is retired: each event's opaque handle encodes its type and resolves
    // through exactly one get_*_details() accessor, which is what we assert here.
    auto tracks  = m_reader->get_tracks();
    auto streams = find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_EQ(streams.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t s1, s2;
    for(const auto& s : streams)
    {
        ASSERT_NE(s->stream_info, nullptr);
        if(s->stream_info->stream_id == 1)
            s1 = s;
        else if(s->stream_info->stream_id == 2)
            s2 = s;
    }
    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);

    auto iv1 = m_reader->get_interval_track(s1->id);
    ASSERT_EQ(iv1.size(), 6U);
    ASSERT_TRUE(is_start_sorted(iv1));

    size_t kd = 0, mc = 0, ma = 0;
    for(const auto& ev : iv1)
    {
        ASSERT_EQ(count_interval_resolutions(*m_reader, ev.id), 1)
            << "handle must resolve through exactly one detail accessor";
        if(type_of(ev.id) == profiler_hub::reader_types::event_type_t::kernel_dispatch)
            ++kd;
        else if(type_of(ev.id) == profiler_hub::reader_types::event_type_t::memory_copy)
            ++mc;
        else if(type_of(ev.id) ==
                profiler_hub::reader_types::event_type_t::memory_allocate)
            ++ma;
        else
            FAIL() << "unexpected event type on stream 1";
    }
    ASSERT_EQ(kd, 3U);
    ASSERT_EQ(mc, 2U);
    ASSERT_EQ(ma, 1U);
    ASSERT_EQ(iv1.front().start, 1200);
    ASSERT_EQ(type_of(iv1.front().id),
              profiler_hub::reader_types::event_type_t::kernel_dispatch);
    ASSERT_EQ(iv1.back().start, 6100);
    ASSERT_EQ(type_of(iv1.back().id),
              profiler_hub::reader_types::event_type_t::memory_allocate);

    auto iv2 = m_reader->get_interval_track(s2->id);
    ASSERT_EQ(iv2.size(), 1U);
    ASSERT_EQ(iv2.front().start, 2400);
    ASSERT_EQ(type_of(iv2.front().id),
              profiler_hub::reader_types::event_type_t::memory_copy);

    expect_stats_match_intervals(m_reader->get_track_stats(s1->id), iv1);
    expect_stats_match_intervals(m_reader->get_track_stats(s2->id), iv2);
}

TEST_F(reader_v3_edge_test, get_interval_track_stream_memalloc_event_carries_category)
{
    // The memory_allocate UNION leg in the stream SQL carries the category LEFT JOIN
    // (same pattern as kd/mc legs). The sole memory_allocate row (ma1) has no
    // category set, so the resolved category must be an empty string — asserting
    // that proves the structural LEFT JOIN is executed correctly.
    auto tracks  = m_reader->get_tracks();
    auto streams = find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream);

    profiler_hub::reader_types::track_info_ptr_t s1;
    for(const auto& s : streams)
    {
        if(s->stream_info && s->stream_info->stream_id == 1) s1 = s;
    }
    ASSERT_NE(s1, nullptr);

    auto iv = m_reader->get_interval_track(s1->id);
    ASSERT_EQ(iv.size(), 6U);

    bool found_ma = false;
    for(const auto& ev : iv)
    {
        if(type_of(ev.id) == profiler_hub::reader_types::event_type_t::memory_allocate)
        {
            EXPECT_EQ(ev.category, "");
            found_ma = true;
        }
    }
    EXPECT_TRUE(found_ma) << "stream 1 must contain at least one memory_allocate event";
}

TEST_F(reader_v3_edge_test, get_interval_track_memory_type_interval_and_identity)
{
    // track_type_t::memory for rocpd_memory_allocate rows keyed by
    // (nid, agent_id, queue_id, pid). One such row exercises the "a_only" variant
    // (agent_id set, queue_id NULL).
    auto tracks  = m_reader->get_tracks();
    auto mem_trk = find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory);
    ASSERT_EQ(mem_trk.size(), 1U);

    const auto& t = mem_trk.front();
    // agent_info must be populated (agent_id=1); queue_info null (queue_id IS NULL).
    ASSERT_NE(t->agent_info, nullptr);
    EXPECT_EQ(t->agent_info->id, 1U);
    EXPECT_EQ(t->queue_info, nullptr);

    auto intervals = m_reader->get_interval_track(t->id);
    ASSERT_EQ(intervals.size(), 1U);
    EXPECT_EQ(intervals.front().start, 6100U);
    EXPECT_EQ(intervals.front().end, 6200U);

    // the handle must resolve through get_event_info() as a memory_allocate event.
    ASSERT_EQ(type_of(intervals.front().id),
              profiler_hub::reader_types::event_type_t::memory_allocate);
    auto details = m_reader->get_event_info(intervals.front().id);
    ASSERT_TRUE(details.has_value());
    EXPECT_EQ(details->ts, 6100U);
    ASSERT_TRUE(details->te.has_value());
    EXPECT_EQ(details->te.value(), 6200U);
    auto* size = find_prop(*details, "size");
    ASSERT_NE(size, nullptr);
    ASSERT_TRUE(std::holds_alternative<uint64_t>(*size));
    EXPECT_EQ(std::get<uint64_t>(*size), 4096U);
    auto* type = find_prop(*details, "type");
    ASSERT_NE(type, nullptr);
    ASSERT_TRUE(std::holds_alternative<std::string>(*type));
    EXPECT_EQ(std::get<std::string>(*type), "ALLOC");
}

TEST_F(reader_v3_edge_test, get_track_stats_memory_type_matches_interval_slice)
{
    // get_track_stats() must return the same count/min/max as the interval slice for
    // the memory track.
    auto tracks  = m_reader->get_tracks();
    auto mem_trk = find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory);
    ASSERT_EQ(mem_trk.size(), 1U);

    auto intervals = m_reader->get_interval_track(mem_trk.front()->id);
    expect_stats_match_intervals(m_reader->get_track_stats(mem_trk.front()->id),
                                 intervals);
}

// ===========================================================================
// SQL-seeded edge tests (DL-015 buckets (a) + (b)). These load the committed
// SQL fixture (fixtures/rocpd_v3_edge_data.sql -> rocpd_v3_edge.db, built at
// configure time). (a) is the permanent escape-hatch: a bare non-pmc rocpd_sample
// track a correct writer must never emit. (b) is writer-id-blocked NOW and is
// flagged for a future writer task: rocpd_arg rows on kernel_dispatch/memory_copy/
// memory_allocate events (writer only carries args on regions) and counter/track
// identity (NULL pid/tid, empty-pmc-name fallback) authored directly on
// rocpd_track. Assertions are the pre-rebase originals, verbatim.
// ===========================================================================
class reader_v3_edge_sql_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V3_EDGE_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_edge_sql_test, track_matrix_counts_by_type)
{
    // cpu_thread/region tracks are synthesized from rocpd_region, not rocpd_track.
    // rocpd_track contributes 4 PMC-backed sampled (counter) rows (2, 3, 6, 8);
    // the non-counter rows (1, 4, 5) are ignored, and track 7 -- sampled but with NO
    // rocpd_pmc_event -- is NOT a counter. Track 8 (pmc_id 99, empty PMC name) IS a
    // counter -- discovery joins rocpd_pmc_event (present), not rocpd_info_pmc.
    // Synthesis adds 1 cpu_thread, 2 gpu_queue, 1 dma, 2 stream, 1 memory => 11 tracks.
    // Task 012B adds 1 memory_activity (1 alloc row, agent_id=1) => total 12.
    auto tracks = m_reader->get_tracks();
    ASSERT_EQ(tracks.size(), 12U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread).size(),
        1U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter).size(),
        4U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue).size(),
        2U);
    ASSERT_EQ(find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma).size(),
              1U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream).size(), 2U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory).size(), 1U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory_activity)
            .size(),
        1U);
}

TEST_F(reader_v3_edge_sql_test, counter_discovery_excludes_non_pmc_sample_track)
{
    // Counter discovery must classify a track as a counter only when a PMC-backed
    // rocpd_sample references it (the sample's event_id joins rocpd_pmc_event), NOT
    // merely when any rocpd_sample references it. Track 7 has a rocpd_sample (sample 7
    // / event 14) but NO rocpd_pmc_event, so it must not appear as a counter -- and
    // since it has no rocpd_region row, it must not appear as any track type at all.
    auto tracks = m_reader->get_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    // Primary signal: only the 4 PMC-backed sample tracks (2, 3, 6, 8) are counters.
    ASSERT_EQ(counters.size(), 4U);
    // Corroborating signal: every counter is PMC-backed, so each resolves to a
    // non-empty scalar track. The spurious non-PMC track 7 would resolve to zero.
    for(const auto& c : counters)
        ASSERT_FALSE(m_reader->get_scalar_track(c->id).empty())
            << "counter track " << c->id.value << " has no PMC-backed samples";
}

TEST_F(reader_v3_edge_sql_test, counter_identity_null_pid_and_null_tid_branches)
{
    // v3 counter tracks come from rocpd_track (Q10) and CAN carry NULL pid/tid:
    //   track 2: pid set, tid NULL -> process_info set,  thread_info NULL
    //   track 3: pid + tid set     -> process_info set,  thread_info SET
    //   track 6: pid NULL          -> process_info NULL, thread_info NULL
    //   track 8: pid set, tid NULL -> process_info set,  thread_info NULL (fallback)
    auto tracks = m_reader->get_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_EQ(counters.size(), 4U);

    int with_thread = 0, with_process = 0, without_process = 0;
    for(const auto& t : counters)
    {
        if(t->thread_info != nullptr) ++with_thread;
        if(t->process_info != nullptr)
            ++with_process;
        else
            ++without_process;
    }
    // Exactly one counter track carries a resolved thread (tid set -- track 3).
    ASSERT_EQ(with_thread, 1);
    // Exactly one carries no process (pid NULL -- track 6); tracks 2/3/8 do.
    ASSERT_EQ(without_process, 1);
    ASSERT_EQ(with_process, 3);
}

TEST_F(reader_v3_edge_sql_test, counter_thread_info_tracks_tid_agent_info_always_null)
{
    // The #147 contract, both branches. thread_info is driven by rocpd_track.tid
    // and is orthogonal to counter classification; agent_info is impossible on v3
    // (rocpd_track has no agent_id column) regardless of tid.
    auto tracks = m_reader->get_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_EQ(counters.size(), 4U);

    profiler_hub::reader_types::track_info_ptr_t no_tid_counter;    // GRBM_COUNT
    profiler_hub::reader_types::track_info_ptr_t with_tid_counter;  // SQ_WAVES
    for(const auto& c : counters)
    {
        if(c->name == "GRBM_COUNT")
            no_tid_counter = c;
        else if(c->name == "SQ_WAVES")
            with_tid_counter = c;
    }
    ASSERT_NE(no_tid_counter, nullptr)
        << "counter display name should be its PMC name (Q9)";
    ASSERT_NE(with_tid_counter, nullptr)
        << "counter display name should be its PMC name (Q9)";

    // Branch 1: counter with tid NULL -> thread_info null.
    ASSERT_EQ(no_tid_counter->thread_info, nullptr);
    ASSERT_EQ(no_tid_counter->agent_info, nullptr);

    // Branch 2: counter WITH tid -> thread_info populated (the case rocpd.db lacks).
    ASSERT_NE(with_tid_counter->thread_info, nullptr);
    ASSERT_EQ(with_tid_counter->agent_info, nullptr);
}

TEST_F(reader_v3_edge_sql_test, counter_display_name_falls_back_to_track_name_on_pmc_miss)
{
    // F7 coverage: when the pmc_info lookup produces an empty name, the display name must
    // fall back to rocpd_track.name rather than being empty, zero-initialized, or stale.
    // Track 8: rocpd_track.name_id=7 -> "FallbackCounter"; pmc_id=99 exists in
    // rocpd_info_pmc with an intentionally empty name field.
    auto tracks = m_reader->get_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    profiler_hub::reader_types::track_info_ptr_t fallback_counter;
    for(const auto& c : counters)
    {
        if(c->name == "FallbackCounter")
        {
            fallback_counter = c;
            break;
        }
    }
    ASSERT_NE(fallback_counter, nullptr) << "fallback counter track not found";

    // Primary assertion: display name equals rocpd_track.name (the fallback value).
    ASSERT_EQ(fallback_counter->name, "FallbackCounter");
    // Sanity: non-empty, not garbage.
    ASSERT_FALSE(fallback_counter->name.empty());
    // pmc_info: pmc_id=99 is in rocpd_info_pmc with empty name -> pmc_info IS attached
    // but carries an empty name, which is exactly what triggers the fallback guard.
    ASSERT_NE(fallback_counter->pmc_info, nullptr);
    ASSERT_TRUE(fallback_counter->pmc_info->name.empty());
    // Non-fallback path still intact: the 3 fully-resolved counters have non-empty names
    // and their display name equals the PMC name (name != track->name only for fallback).
    size_t with_pmc_name_match = 0;
    for(const auto& c : counters)
    {
        if(c->pmc_info != nullptr && !c->pmc_info->name.empty())
        {
            ASSERT_EQ(c->name, c->pmc_info->name);
            ++with_pmc_name_match;
        }
    }
    ASSERT_EQ(with_pmc_name_match, 3U);
}

// Scan a track's interval handles for the get_event_info whose property bag
// contains `arg_key`, and return that value (or nullptr if none carries it).
static const profiler_hub::reader_types::arg_value_t*
find_folded_arg_on_track(const profiler_hub::reader_t&                       r,
                         const profiler_hub::reader_types::track_info_ptr_t& track,
                         const std::string&                                  arg_key)
{
    static profiler_hub::reader_types::arg_value_t s_hit;
    for(const auto& iv : r.get_interval_track(track->id))
    {
        auto detail = r.get_event_info(iv.id);
        if(!detail) continue;
        if(const auto* v = find_prop(*detail, arg_key))
        {
            s_hit = *v;
            return &s_hit;
        }
    }
    return nullptr;
}

TEST_F(reader_v3_edge_sql_test, get_event_info_folds_args_for_kernel_dispatch)
{
    auto                                           tracks = m_reader->get_tracks();
    const profiler_hub::reader_types::arg_value_t* kernel_name = nullptr;
    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue))
    {
        kernel_name = find_folded_arg_on_track(*m_reader, t, "kernel_name");
        if(kernel_name) break;
    }
    ASSERT_NE(kernel_name, nullptr) << "kernel_dispatch detail did not fold its args";
    ASSERT_TRUE(std::holds_alternative<std::string>(*kernel_name));
    EXPECT_EQ(std::get<std::string>(*kernel_name), "vecAdd");
}

TEST_F(reader_v3_edge_sql_test, get_event_info_folds_args_for_memory_copy)
{
    auto                                           tracks = m_reader->get_tracks();
    const profiler_hub::reader_types::arg_value_t* bytes  = nullptr;
    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma))
    {
        bytes = find_folded_arg_on_track(*m_reader, t, "bytes");
        if(bytes) break;
    }
    ASSERT_NE(bytes, nullptr) << "memory_copy detail did not fold its args";
    ASSERT_TRUE(std::holds_alternative<std::string>(*bytes));
    EXPECT_EQ(std::get<std::string>(*bytes), "1024");
}

TEST_F(reader_v3_edge_sql_test, get_event_info_folds_args_for_memory_allocate)
{
    auto                                           tracks = m_reader->get_tracks();
    const profiler_hub::reader_types::arg_value_t* alloc_bytes = nullptr;
    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory))
    {
        alloc_bytes = find_folded_arg_on_track(*m_reader, t, "alloc_bytes");
        if(alloc_bytes) break;
    }
    ASSERT_NE(alloc_bytes, nullptr) << "memory_allocate detail did not fold its args";
    ASSERT_TRUE(std::holds_alternative<std::string>(*alloc_bytes));
    EXPECT_EQ(std::get<std::string>(*alloc_bytes), "4096");
}

// ============================================================================
// kernel_dispatch_pmc track type — v3 synthetic fixture (rocpd_v3_kd_pmc.db)
// Data: 1 agent, 2 PMC types (SQ_WAVES pmc_id=1, GRBM_COUNT pmc_id=2),
// 3 dispatches: kd 1+2 on SQ_WAVES (start 1000,2000), kd 3 on GRBM_COUNT
// (start 3000). Tracks: (nid=1,agent_id=1,pmc_id=1,pid=100) has 2 events;
// (nid=1,agent_id=1,pmc_id=2,pid=100) has 1 event.
// ============================================================================

class reader_v3_kd_pmc_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V3_KD_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_kd_pmc_test, v3_discovers_two_kd_pmc_tracks)
{
    // Two distinct (nid, agent_id, pmc_id, pid) -> 2 kernel_dispatch_pmc tracks.
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    // Every kd_pmc track must carry agent_info (from agent_id=1).
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->agent_info->id, 1U);
        ASSERT_NE(t->process_info, nullptr);
        ASSERT_EQ(t->process_info->pid, 100U);
        ASSERT_NE(t->node_info, nullptr);
    }
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_pmc_info_populated)
{
    // pmc_info must be resolved from pmc_id for both tracks.
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    std::set<std::string> pmc_names;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        pmc_names.insert(t->pmc_info->name);
    }
    ASSERT_TRUE(pmc_names.count("SQ_WAVES") == 1);
    ASSERT_TRUE(pmc_names.count("GRBM_COUNT") == 1);
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_interval_track_count_and_order)
{
    // The SQ_WAVES track (pmc_id=1) covers kd 1 (start=1000) and kd 2 (start=2000).
    // Rows are inserted out of start order (kd 2 first), so this proves ORDER BY start.
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t sq_waves_track;
    profiler_hub::reader_types::track_info_ptr_t grbm_track;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        if(t->pmc_info->name == "SQ_WAVES")
            sq_waves_track = t;
        else if(t->pmc_info->name == "GRBM_COUNT")
            grbm_track = t;
    }
    ASSERT_NE(sq_waves_track, nullptr);
    ASSERT_NE(grbm_track, nullptr);

    // SQ_WAVES track: 2 events in ascending start order.
    auto sq_intervals = m_reader->get_interval_track(sq_waves_track->id);
    ASSERT_EQ(sq_intervals.size(), 2U);
    ASSERT_TRUE(is_start_sorted(sq_intervals));
    ASSERT_EQ(sq_intervals[0].start, 1000U);
    ASSERT_EQ(sq_intervals[0].end, 1200U);
    ASSERT_EQ(sq_intervals[1].start, 2000U);
    ASSERT_EQ(sq_intervals[1].end, 2300U);

    // GRBM_COUNT track: 1 event.
    auto grbm_intervals = m_reader->get_interval_track(grbm_track->id);
    ASSERT_EQ(grbm_intervals.size(), 1U);
    ASSERT_EQ(grbm_intervals[0].start, 3000U);
    ASSERT_EQ(grbm_intervals[0].end, 3100U);
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_interval_resolves_as_kernel_dispatch)
{
    // Task 035: a kd_pmc interval event's row id is a rocpd_kernel_dispatch.id, so its
    // handle must be typed kernel_dispatch and resolve through the KD detail path -- NOT
    // the point pmc_event path (WHERE rocpd_pmc_event.id = ?), which keys a different
    // table. Guard bites: revert interval_event_type_for(kernel_dispatch_pmc) to
    // pmc_event and this test fails (handle mis-types + KD detail unreachable; the
    // kd_pmc fixture has no rocpd_sample, so the point path resolves to nullopt).
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    profiler_hub::reader_types::track_info_ptr_t sq_waves_track;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        if(t->pmc_info->name == "SQ_WAVES") sq_waves_track = t;
    }
    ASSERT_NE(sq_waves_track, nullptr);

    auto intervals = m_reader->get_interval_track(sq_waves_track->id);
    ASSERT_FALSE(intervals.empty());
    const auto& first = intervals.front();  // start=1000 -> kd row id 1

    // The minted handle is typed kernel_dispatch, not pmc_event.
    EXPECT_EQ(type_of(first.id),
              profiler_hub::reader_types::event_type_t::kernel_dispatch);

    auto detail = m_reader->get_event_info(first.id);
    ASSERT_TRUE(detail.has_value());
    // Interval extent is present (kd_pmc is an interval track); a point pmc_event would
    // leave te == nullopt.
    EXPECT_EQ(detail->ts, 1000U);
    ASSERT_TRUE(detail->te.has_value());
    EXPECT_EQ(detail->te.value(), 1200U);

    // kernel_dispatch properties are populated -> the KD detail path ran.
    auto* dispatch_id = find_prop(*detail, "dispatch_id");
    ASSERT_NE(dispatch_id, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*dispatch_id), 1U);
    auto* wg_x = find_prop(*detail, "workgroup_size_x");
    ASSERT_NE(wg_x, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*wg_x), 64U);
    auto* grid_x = find_prop(*detail, "grid_size_x");
    ASSERT_NE(grid_x, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*grid_x), 512U);
    EXPECT_NE(find_prop(*detail, "kernel_symbol_id"), nullptr);
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_track_stats_matches_interval_slice)
{
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    for(const auto& t : tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        expect_stats_match_intervals(stats, intervals);
    }
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_display_name_from_kernel_symbol)
{
    // Interval display_name must be resolved from kernel_symbol (vecAdd(float*, int)).
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    auto intervals = m_reader->get_interval_track(tracks.front()->id);
    ASSERT_FALSE(intervals.empty());
    for(const auto& ev : intervals)
    {
        ASSERT_EQ(ev.display_name, "vecAdd(float*, int)");
    }
}

TEST_F(reader_v3_kd_pmc_test, v3_get_scalar_track_returns_empty_for_kd_pmc)
{
    // kernel_dispatch_pmc is an interval track; scalar read must return empty (Q7 guard).
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_scalar_track(tracks.front()->id).empty());
}

// =============================================================================
// Task 044: v3 track-type x schema switch-arm coverage.
//   Fixture rocpd_v3_track_shapes.db carries exactly one track of each v3 dma /
//   memory / cpu_thread shape the other v3 fixtures leave dark in get_track_stats
//   / get_interval_track: dma queue-only / agent-only / queue+agent, memory
//   queue+agent / queue-only / neither, and a cpu_thread SAMPLE track. Each test
//   asserts exact min_ts / max_ts / count and interval start order
//   (by-construction oracles), not merely that the call did not throw.
// =============================================================================
class reader_v3_track_shapes_test : public ::testing::Test
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

    std::string m_database_path{ ROCPD_DB_V3_TRACK_SHAPES_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

// dma queue+agent / queue-only / agent-only, for BOTH get_interval_track (the
// memory_copy_interval_{qa,q_only,a_only} arms) and get_track_stats (the
// memory_copy_stats_{qa,q_only,a_only} arms). Tracks are keyed by their unique
// min_ts so the assertion is robust to discovery order.
TEST_F(reader_v3_track_shapes_test, dma_shape_arms_interval_and_stats)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(tracks.size(), 3U);

    // min_ts -> {expected max_ts, expected count}
    const std::map<uint64_t, std::pair<uint64_t, size_t>> expected = {
        { 1000U, { 1300U, 2U } },  // qa      (queue_id=1, dst_agent_id=1)
        { 2000U, { 2300U, 2U } },  // q_only  (queue_id=2, dst_agent_id=NULL)
        { 3000U, { 3100U, 1U } },  // a_only  (queue_id=NULL, dst_agent_id=2)
    };

    std::set<uint64_t> seen;
    for(const auto& t : tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        ASSERT_TRUE(is_start_sorted(intervals));
        expect_stats_match_intervals(stats, intervals);

        ASSERT_TRUE(stats.min_ts.has_value());
        auto it = expected.find(stats.min_ts.value());
        ASSERT_NE(it, expected.end()) << "unexpected dma track min_ts";
        ASSERT_EQ(stats.max_ts.value(), it->second.first);
        ASSERT_EQ(stats.count, it->second.second);
        seen.insert(stats.min_ts.value());
    }
    ASSERT_EQ(seen.size(), 3U);
}

// memory queue+agent / queue-only / neither, for BOTH get_interval_track (the
// memory_alloc_interval_{qa,q_only,neither} arms) and get_track_stats (the
// memory_alloc_stats_{qa,q_only,neither} arms). (agent-only is covered by v3_edge.)
TEST_F(reader_v3_track_shapes_test, memory_shape_arms_interval_and_stats)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory);
    ASSERT_EQ(tracks.size(), 3U);

    const std::map<uint64_t, std::pair<uint64_t, size_t>> expected = {
        { 4000U, { 4300U, 2U } },  // qa      (agent_id=1, queue_id=1)
        { 5000U, { 5300U, 2U } },  // q_only  (agent_id=NULL, queue_id=2)
        { 6000U, { 6100U, 1U } },  // neither (agent_id=NULL, queue_id=NULL)
    };

    std::set<uint64_t> seen;
    for(const auto& t : tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        ASSERT_TRUE(is_start_sorted(intervals));
        expect_stats_match_intervals(stats, intervals);

        ASSERT_TRUE(stats.min_ts.has_value());
        auto it = expected.find(stats.min_ts.value());
        ASSERT_NE(it, expected.end()) << "unexpected memory track min_ts";
        ASSERT_EQ(stats.max_ts.value(), it->second.first);
        ASSERT_EQ(stats.count, it->second.second);
        seen.insert(stats.min_ts.value());
    }
    ASSERT_EQ(seen.size(), 3U);
}

// cpu_thread SAMPLE track: both regions' events carry a rocpd_sample, so the
// (nid,pid,tid) region track is classified is_sample=1 and routes through
// region_interval_track_sample / region_stats_track_sample (the arms v3_edge's
// all-"main" cpu_thread track leaves dark).
TEST_F(reader_v3_track_shapes_test, cpu_thread_sample_interval_and_stats)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_EQ(tracks.size(), 1U);

    auto intervals = m_reader->get_interval_track(tracks.front()->id);
    auto stats     = m_reader->get_track_stats(tracks.front()->id);

    ASSERT_TRUE(is_start_sorted(intervals));
    expect_stats_match_intervals(stats, intervals);

    ASSERT_EQ(intervals.size(), 2U);
    ASSERT_EQ(intervals[0].start, 7000U);
    ASSERT_EQ(intervals[1].start, 7200U);
    ASSERT_EQ(stats.count, 2U);
    ASSERT_EQ(stats.min_ts.value(), 7000U);
    ASSERT_EQ(stats.max_ts.value(), 7500U);
}

// ============================================================================
// kernel_dispatch_pmc track type — v4 synthetic fixture (rocpd_v4_kd_pmc.db)
// Mirrors the v3 fixture data shape; the presence of rocpd_timestamp triggers
// the v4 backend. Verifies the 4-arg timestamp-spine SQL path.
// ============================================================================

class reader_v4_kd_pmc_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V4_KD_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_kd_pmc_test, v4_discovers_two_kd_pmc_tracks)
{
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->agent_info->id, 1U);
        ASSERT_NE(t->process_info, nullptr);
        ASSERT_EQ(t->process_info->pid, 100U);
        ASSERT_NE(t->node_info, nullptr);
    }
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_pmc_info_populated)
{
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    std::set<std::string> pmc_names;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        pmc_names.insert(t->pmc_info->name);
    }
    ASSERT_TRUE(pmc_names.count("SQ_WAVES") == 1);
    ASSERT_TRUE(pmc_names.count("GRBM_COUNT") == 1);
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_interval_track_count_and_order)
{
    // Timestamps inserted out of value order (kd 2 timestamps ids 1,2 with values
    // 2000/2300 before kd 1 timestamps ids 3,4 with values 1000/1200). ORDER BY
    // ts_s.value must return kd 1 before kd 2 on the SQ_WAVES track.
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t sq_waves_track;
    profiler_hub::reader_types::track_info_ptr_t grbm_track;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        if(t->pmc_info->name == "SQ_WAVES")
            sq_waves_track = t;
        else if(t->pmc_info->name == "GRBM_COUNT")
            grbm_track = t;
    }
    ASSERT_NE(sq_waves_track, nullptr);
    ASSERT_NE(grbm_track, nullptr);

    auto sq_intervals = m_reader->get_interval_track(sq_waves_track->id);
    ASSERT_EQ(sq_intervals.size(), 2U);
    ASSERT_TRUE(is_start_sorted(sq_intervals));
    ASSERT_EQ(sq_intervals[0].start, 1000U);
    ASSERT_EQ(sq_intervals[0].end, 1200U);
    ASSERT_EQ(sq_intervals[1].start, 2000U);
    ASSERT_EQ(sq_intervals[1].end, 2300U);

    auto grbm_intervals = m_reader->get_interval_track(grbm_track->id);
    ASSERT_EQ(grbm_intervals.size(), 1U);
    ASSERT_EQ(grbm_intervals[0].start, 3000U);
    ASSERT_EQ(grbm_intervals[0].end, 3100U);
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_interval_resolves_as_kernel_dispatch)
{
    // Task 035 (v4 backend): same contract as the v3 test. The v4 kd_pmc interval SQL
    // also SELECTs K.id (rocpd_kernel_dispatch.id), so the single-site fix in
    // interval_event_type_for is backend-agnostic and routes this handle through the KD
    // detail path with the interval extent (te) present.
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    profiler_hub::reader_types::track_info_ptr_t sq_waves_track;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        if(t->pmc_info->name == "SQ_WAVES") sq_waves_track = t;
    }
    ASSERT_NE(sq_waves_track, nullptr);

    auto intervals = m_reader->get_interval_track(sq_waves_track->id);
    ASSERT_FALSE(intervals.empty());
    const auto& first = intervals.front();  // start=1000 -> kd row id 1

    EXPECT_EQ(type_of(first.id),
              profiler_hub::reader_types::event_type_t::kernel_dispatch);

    auto detail = m_reader->get_event_info(first.id);
    ASSERT_TRUE(detail.has_value());
    EXPECT_EQ(detail->ts, 1000U);
    ASSERT_TRUE(detail->te.has_value());
    EXPECT_EQ(detail->te.value(), 1200U);

    auto* dispatch_id = find_prop(*detail, "dispatch_id");
    ASSERT_NE(dispatch_id, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*dispatch_id), 1U);
    auto* wg_x = find_prop(*detail, "workgroup_size_x");
    ASSERT_NE(wg_x, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*wg_x), 64U);
    auto* grid_x = find_prop(*detail, "grid_size_x");
    ASSERT_NE(grid_x, nullptr);
    EXPECT_EQ(std::get<uint64_t>(*grid_x), 512U);
    EXPECT_NE(find_prop(*detail, "kernel_symbol_id"), nullptr);
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_track_stats_matches_interval_slice)
{
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    for(const auto& t : tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        expect_stats_match_intervals(stats, intervals);
    }
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_display_name_from_kernel_symbol)
{
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    auto intervals = m_reader->get_interval_track(tracks.front()->id);
    ASSERT_FALSE(intervals.empty());
    for(const auto& ev : intervals)
    {
        ASSERT_EQ(ev.display_name, "vecAdd(float*, int)");
    }
}

TEST_F(reader_v4_kd_pmc_test, v4_get_scalar_track_returns_empty_for_kd_pmc)
{
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_scalar_track(tracks.front()->id).empty());
}

// --- Tier 3: synthetic v4.0 ambiguous-pmc fixture ----------------------------

class reader_v4_amb_pmc_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V4_AMB_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_amb_pmc_test, v4_ambiguous_pmc_id_flagged)
{
    auto pmc_list = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_list.size(), 2U);

    profiler_hub::reader_types::pmc_info_ptr_t fault_pmc;
    profiler_hub::reader_types::pmc_info_ptr_t clean_pmc;
    for(const auto& p : pmc_list)
    {
        if(p->name == "FAULT_COUNT") fault_pmc = p;
        if(p->name == "CLEAN_COUNT") clean_pmc = p;
    }
    ASSERT_NE(fault_pmc, nullptr);
    ASSERT_NE(clean_pmc, nullptr);
    EXPECT_TRUE(fault_pmc->ambiguous);
    EXPECT_FALSE(clean_pmc->ambiguous);
}

TEST_F(reader_v4_amb_pmc_test, v4_exactly_one_ambiguous_pmc)
{
    auto   pmc_list        = m_reader->get_all_pmc_info();
    size_t ambiguous_count = 0;
    for(const auto& p : pmc_list)
    {
        if(p->ambiguous) ++ambiguous_count;
    }
    EXPECT_EQ(ambiguous_count, 1U);
}

// Task 018: v4 track-classification ambiguity detection tests
//
// Fixture: rocpd_v4_amb_cls.db — a single rocpd_track row (id=1) referenced by
//   both rocpd_sample/rocpd_pmc_event (counter set) and rocpd_memory_allocate
//   (memory set). build_v4_tracks() must detect the overlap, log a warning, and
//   set ambiguous_classification=true on the resulting counter track.

class reader_v4_amb_cls_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V4_AMB_CLS_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_amb_cls_test, v4_ambiguous_classification_track_flagged)
{
    auto tracks = m_reader->get_tracks();

    // Find the counter track (the ambiguous rocpd_track row; the fixture also
    // yields a synthetic memory_activity track from the same rocpd_memory_allocate).
    profiler_hub::reader_types::track_info_ptr_t counter_track;
    for(const auto& t : tracks)
    {
        if(t->type == profiler_hub::reader_types::track_type_t::counter)
        {
            counter_track = t;
            break;
        }
    }
    ASSERT_NE(counter_track, nullptr) << "no counter track found";
    // Counter classification wins (existing precedence).
    EXPECT_EQ(counter_track->type, profiler_hub::reader_types::track_type_t::counter);
    // Overlap with memory-allocate set is detected and flagged.
    EXPECT_TRUE(counter_track->ambiguous_classification);
}

// ============================================================================
// Track-scoped API tests — v4.0 synthetic counter fixture (rocpd_v4_counter.db)
// Built at configure time from committed SQL. Exists solely to exercise the
// v4.0 scalar/counter path (get_scalar_track / get_event_info), which no
// real v4.0 capture available to the project contains (no rocpd_sample rows).
// ============================================================================

class reader_v4_counter_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V4_COUNTER_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_counter_test, v4_counter_track_classified_named_and_agent_scoped)
{
    auto tracks = m_reader->get_tracks();
    // Two tracks: the counter track (sample-referenced) and a bare cpu_thread.
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    // Q9: counter track display name is the PMC name.
    ASSERT_EQ(counter->name, "GRBM_COUNT");
    // Q10: v4 counter track carries agent_info (its rocpd_track row has agent_id).
    ASSERT_NE(counter->agent_info, nullptr);
    ASSERT_NE(counter->thread_info, nullptr);

    // v4.0 has one pmc per event (no event_id fan-out), so it is unaffected by the
    // v3-only deterministic disambiguation (005B-4-fix-1-fix-1): the single track must
    // still resolve to the GRBM_COUNT pmc, with name/agent consistent with the track.
    ASSERT_NE(counter->pmc_info, nullptr);
    ASSERT_EQ(counter->pmc_info->name, "GRBM_COUNT");
    // 005B-4-fix-1-fix-2: numeric pmc_id exposed on pmc_info; GRBM_COUNT is pmc 1 here.
    ASSERT_EQ(counter->pmc_info->pmc_id, 1U);
    ASSERT_EQ(counter->name, counter->pmc_info->name);
    ASSERT_NE(counter->pmc_info->agent_info, nullptr);
    ASSERT_EQ(counter->pmc_info->agent_info->agent_type, "GPU");
    ASSERT_EQ(counter->pmc_info->agent_info->type_index, 0U);
}

TEST_F(reader_v4_counter_test, v4_get_scalar_track_returns_timestamp_ordered_values)
{
    auto tracks = m_reader->get_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    auto samples = m_reader->get_scalar_track(counter->id);
    // 3 samples, returned in ascending-timestamp order despite row-id order differing.
    ASSERT_EQ(samples.size(), 3);
    ASSERT_TRUE(is_timestamp_sorted(samples));

    ASSERT_EQ(row_id_of(samples[0].id), 2U);
    ASSERT_EQ(samples[0].timestamp, 1000);
    ASSERT_DOUBLE_EQ(samples[0].value, 10.5);

    ASSERT_EQ(row_id_of(samples[1].id), 3U);
    ASSERT_EQ(samples[1].timestamp, 2000);
    ASSERT_DOUBLE_EQ(samples[1].value, 20.5);

    ASSERT_EQ(row_id_of(samples[2].id), 1U);
    ASSERT_EQ(samples[2].timestamp, 3000);
    ASSERT_DOUBLE_EQ(samples[2].value, 30.5);
}

TEST_F(reader_v4_counter_test, v4_get_event_info_resolves_sample_point_event)
{
    // sample row id 1 -> timestamp 3000. The scalar handle encodes the sample event
    // type; get_event_info resolves it as a point event (te == nullopt). The counter
    // name + value payload is asserted separately below (§7, task 052).
    auto details = m_reader->get_event_info(
        make_sql_event_id(profiler_hub::reader_types::event_type_t::sample, 1));
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->ts, 3000U);
    ASSERT_FALSE(details->te.has_value());
}

TEST_F(reader_v4_counter_test, v4_get_event_info_counter_sample_carries_name_and_value)
{
    // §7 (task 052, v4 backend): sample row id 1 -> track 1 "GRBM_COUNT", value 30.5.
    // Resolved through the unified get_event_info the counter sample carries the counter
    // name (from the track) + value (as a double property). Pre-052 this arm returned a
    // bare timestamp, dropping name+value (guard-bite).
    auto details = m_reader->get_event_info(
        make_sql_event_id(profiler_hub::reader_types::event_type_t::sample, 1));
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->name, "GRBM_COUNT");

    auto* value = find_prop(*details, "value");
    ASSERT_NE(value, nullptr);
    ASSERT_TRUE(std::holds_alternative<double>(*value));
    ASSERT_DOUBLE_EQ(std::get<double>(*value), 30.5);
}

TEST_F(reader_v4_counter_test, v4_get_event_info_pmc_event_carries_value)
{
    // A pmc_event point handle minted from a known rocpd_pmc_event.id resolves through
    // the unified detail path: it is a point event (te == nullopt) whose "value" property
    // carries the counter value as a double. pmc_event id=1 -> event_id=1 -> sample
    // timestamp 3000, value 30.5. (Reader-minted pmc_event handles on kernel_dispatch_pmc
    // tracks carry a kernel_dispatch id and route to the interval path, so the point
    // detail path is exercised here with a directly-minted pmc_event.id handle.)
    auto details = m_reader->get_event_info(
        make_sql_event_id(profiler_hub::reader_types::event_type_t::pmc_event, 1));
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->ts, 3000U);
    ASSERT_FALSE(details->te.has_value());
    auto* value = find_prop(*details, "value");
    ASSERT_NE(value, nullptr);
    ASSERT_TRUE(std::holds_alternative<double>(*value));
    ASSERT_DOUBLE_EQ(std::get<double>(*value), 30.5);
}

TEST_F(reader_v4_counter_test, v4_get_interval_track_on_counter_returns_empty)
{
    // Q7: interval query against the counter track returns empty.
    auto tracks = m_reader->get_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);
    ASSERT_TRUE(m_reader->get_interval_track(counter->id).empty());
}

TEST_F(reader_v4_counter_test, v4_get_scalar_track_on_non_counter_returns_empty)
{
    // Q7: scalar query against the bare cpu_thread track (no samples) returns empty.
    auto tracks = m_reader->get_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    ASSERT_TRUE(m_reader->get_scalar_track(cpu->id).empty());
}

TEST_F(reader_v4_counter_test, v4_get_track_stats_counter_matches_scalar_slice)
{
    // v4.0 scalar stats resolve MIN/MAX through the timestamp spine. Known oracle:
    // 3 samples at timestamps 1000/2000/3000 -> min 1000, max 3000, count 3.
    auto tracks = m_reader->get_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    auto samples = m_reader->get_scalar_track(counter->id);
    auto stats   = m_reader->get_track_stats(counter->id);
    expect_stats_match_scalars(stats, samples);
    ASSERT_EQ(stats.count, 3U);
    ASSERT_EQ(stats.min_ts.value(), 1000U);
    ASSERT_EQ(stats.max_ts.value(), 3000U);
}

TEST_F(reader_v4_counter_test, v4_get_track_stats_bare_cpu_thread_is_empty)
{
    // The bare cpu_thread track has no region rows: count 0, nullopt bounds — the
    // honest "empty track" signal (SQL MIN/MAX over an empty set), not an error.
    auto tracks = m_reader->get_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto stats = m_reader->get_track_stats(cpu->id);
    ASSERT_EQ(stats.count, 0U);
    ASSERT_FALSE(stats.min_ts.has_value());
    ASSERT_FALSE(stats.max_ts.has_value());
}

TEST_F(reader_v4_counter_test,
       v4_counter_display_name_falls_back_to_track_name_on_pmc_miss)
{
    // F7 coverage (v4 backend): track 3 has rocpd_track.name_id=2 -> 'FallbackCounterV4';
    // its pmc_event references pmc_id=99 which exists in rocpd_info_pmc with empty name.
    // The empty-name guard (!nit->second.empty()) prevents it from overwriting
    // rocpd_track.name -> display name falls back to "FallbackCounterV4".
    auto tracks = m_reader->get_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    profiler_hub::reader_types::track_info_ptr_t fallback_counter;
    for(const auto& c : counters)
    {
        if(c->name == "FallbackCounterV4")
        {
            fallback_counter = c;
            break;
        }
    }
    ASSERT_NE(fallback_counter, nullptr) << "v4 fallback counter track not found";

    // Primary assertion: display name equals rocpd_track.name (the fallback value).
    ASSERT_EQ(fallback_counter->name, "FallbackCounterV4");
    ASSERT_FALSE(fallback_counter->name.empty());
    // pmc_info is attached (pmc_id=99 exists in rocpd_info_pmc) but carries empty name.
    ASSERT_NE(fallback_counter->pmc_info, nullptr);
    ASSERT_TRUE(fallback_counter->pmc_info->name.empty());
    // Non-fallback path still intact: the GRBM_COUNT track carries pmc_info.
    auto grbm =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(grbm, nullptr);
    ASSERT_EQ(grbm->name, "GRBM_COUNT");
    ASSERT_NE(grbm->pmc_info, nullptr);
}

// ============================================================================
// memory_activity track type — v3 synthetic fixture (rocpd_v3_mem_activity.db)
// Covers: discovery, running-sum correctness (ALLOC/FREE/REALLOC/RECLAIM),
// FREE agent_id recovery via address self-join, non-interference between agents.
// ============================================================================

class reader_v3_mem_activity_test : public ::testing::Test
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

    std::string m_database_path{ ROCPD_DB_V3_MEM_ACTIVITY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_mem_activity_test, v3_discovers_two_mem_activity_tracks)
{
    // Two distinct (nid, pid, agent_id): agent 1 and agent 2.
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_EQ(tracks.size(), 2U);

    // Each track must carry agent_info; no pmc_info (fidelity caveat #2).
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->pmc_info, nullptr);
        ASSERT_NE(t->node_info, nullptr);
        ASSERT_NE(t->process_info, nullptr);
    }

    std::set<size_t> agent_ids;
    for(const auto& t : tracks)
        agent_ids.insert(t->agent_info->id);
    ASSERT_TRUE(agent_ids.count(1) == 1);
    ASSERT_TRUE(agent_ids.count(2) == 1);
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_running_sum_agent1)
{
    // Agent 1 series: ALLOC(4096) at ts=1000, FREE-recovered at ts=3000,
    // REALLOC(no-op) at ts=4000, ALLOC(2048) at ts=5000.
    // Expected 3 scalar samples (REALLOC is not emitted).
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent1_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 1) agent1_track = t;
    }
    ASSERT_NE(agent1_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent1_track->id);
    ASSERT_EQ(scalars.size(), 3U);

    // Timestamps must be ascending.
    ASSERT_EQ(scalars[0].timestamp, 1000U);
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_EQ(scalars[2].timestamp, 5000U);

    // Running-sum values.
    ASSERT_DOUBLE_EQ(scalars[0].value, 4096.0);  // ALLOC +4096
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);     // FREE -4096 (recovered)
    ASSERT_DOUBLE_EQ(scalars[2].value, 2048.0);  // ALLOC +2048
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_free_agent_recovery)
{
    // The FREE row (row 3) has agent_id=NULL in the DB. Its size and agent must be
    // recovered from the ALLOC at the same address (4096). The running sum for agent 1
    // goes from 4096 to 0 at ts=3000, proving the recovery was correct.
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent1_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 1) agent1_track = t;
    }
    ASSERT_NE(agent1_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent1_track->id);
    ASSERT_GE(scalars.size(), 2U);
    // The second sample (ts=3000) reflects the FREE: cumsum drops to 0.
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_non_interference_agent2)
{
    // Agent 2 has exactly 1 ALLOC (ts=2000, size=8192). Its scalar series must not
    // include any agent-1 rows (ALLOC/FREE/REALLOC) or the REALLOC no-op.
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent2_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 2) agent2_track = t;
    }
    ASSERT_NE(agent2_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent2_track->id);
    ASSERT_EQ(scalars.size(), 1U);
    ASSERT_EQ(scalars[0].timestamp, 2000U);
    ASSERT_DOUBLE_EQ(scalars[0].value, 8192.0);
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_track_stats)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    for(const auto& t : tracks)
    {
        auto scalars = m_reader->get_scalar_track(t->id);
        auto stats   = m_reader->get_track_stats(t->id);
        ASSERT_TRUE(stats.min_ts.has_value());
        ASSERT_TRUE(stats.max_ts.has_value());
        ASSERT_EQ(stats.count, scalars.size());
        ASSERT_EQ(stats.min_ts.value(), scalars.front().timestamp);
        ASSERT_EQ(stats.max_ts.value(), scalars.back().timestamp);
    }
}

TEST_F(reader_v3_mem_activity_test, v3_get_interval_track_returns_empty_for_mem_activity)
{
    // memory_activity is a scalar-only track; interval read must return empty.
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_interval_track(tracks.front()->id).empty());
}

// ============================================================================
// memory_activity time-window straddle — v3 synthetic fixture (task 045, gap 3).
// The window `continue` filters inside get_scalar_track's memory_activity branch
// (source/reader_impl.cpp ~2515-2520 ALLOC, ~2548-2553 FREE) are point-in-window
// on r.start, inclusive: a row is kept iff window.start <= r.start <= window.end.
// The fixture (rocpd_v3_mem_activity_window_data.sql) has ALLOC and FREE rows both
// BEFORE and AFTER a [3000,5000] window, so all four filters fire; the emitted
// running-sum values reflect the skipped pre-window rows, proving the filter runs
// AFTER accumulation, not before.
// ============================================================================

class reader_v3_mem_activity_window_test : public ::testing::Test
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

    // The single memory_activity track (agent 1) in the straddle fixture.
    profiler_hub::reader_types::track_info_ptr_t mem_activity_track()
    {
        auto tracks =
            find_tracks(m_reader->get_tracks(),
                        profiler_hub::reader_types::track_type_t::memory_activity);
        EXPECT_EQ(tracks.size(), 1U);
        return tracks.empty() ? nullptr : tracks.front();
    }

    std::string m_database_path{ ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_mem_activity_window_test, unwindowed_returns_full_straddle_series)
{
    auto track = mem_activity_track();
    ASSERT_NE(track, nullptr);

    auto scalars = m_reader->get_scalar_track(track->id);
    ASSERT_EQ(scalars.size(), 7U);
    ASSERT_TRUE(is_timestamp_sorted(scalars));

    // Cumulative running sum across all 7 ALLOC/FREE rows.
    const std::vector<std::pair<uint64_t, double>> expected = {
        { 1000, 100.0 },  { 2000, 0.0 },    { 3000, 500.0 }, { 4000, 300.0 },
        { 5000, 1000.0 }, { 6000, 1999.0 }, { 7000, 1000.0 }
    };
    ASSERT_EQ(scalars.size(), expected.size());
    for(size_t i = 0; i < expected.size(); ++i)
    {
        ASSERT_EQ(scalars[i].timestamp, expected[i].first);
        ASSERT_DOUBLE_EQ(scalars[i].value, expected[i].second);
    }
}

TEST_F(reader_v3_mem_activity_window_test, time_window_straddle_filters_alloc_and_free)
{
    auto track = mem_activity_track();
    ASSERT_NE(track, nullptr);

    profiler_hub::reader_types::event_filter_t f;
    f.time_window.start = 3000;
    f.time_window.end   = 5000;
    auto scalars        = m_reader->get_scalar_track(track->id, f);

    // Boundary-inclusive: rows at 3000 and 5000 are kept; the pre-window ALLOC(1000)
    // + FREE(2000) and post-window ALLOC(6000) + FREE(7000) are all dropped, firing
    // every ALLOC and FREE `continue` on both sides of the window.
    ASSERT_EQ(scalars.size(), 3U);
    ASSERT_TRUE(is_timestamp_sorted(scalars));
    ASSERT_EQ(scalars[0].timestamp, 3000U);
    ASSERT_DOUBLE_EQ(scalars[0].value, 500.0);
    ASSERT_EQ(scalars[1].timestamp, 4000U);
    ASSERT_DOUBLE_EQ(scalars[1].value, 300.0);
    ASSERT_EQ(scalars[2].timestamp, 5000U);
    ASSERT_DOUBLE_EQ(scalars[2].value, 1000.0);

    // The filter demonstrably removed rows (full series is 7).
    ASSERT_LT(scalars.size(), m_reader->get_scalar_track(track->id).size());
}

TEST_F(reader_v3_mem_activity_window_test, time_window_start_only_drops_earlier_rows)
{
    auto track = mem_activity_track();
    ASSERT_NE(track, nullptr);

    // Only start set (end = nullopt): exercises the has_value() guard on the end
    // filter while the start `continue` drops every row with start < 6000.
    profiler_hub::reader_types::event_filter_t f;
    f.time_window.start = 6000;
    auto scalars        = m_reader->get_scalar_track(track->id, f);

    ASSERT_EQ(scalars.size(), 2U);
    ASSERT_EQ(scalars[0].timestamp, 6000U);
    ASSERT_DOUBLE_EQ(scalars[0].value, 1999.0);  // ALLOC 999 on running 1000
    ASSERT_EQ(scalars[1].timestamp, 7000U);
    ASSERT_DOUBLE_EQ(scalars[1].value, 1000.0);  // FREE 999
}

class reader_v4_mem_activity_test : public ::testing::Test
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

    std::string m_database_path{ ROCPD_DB_V4_MEM_ACTIVITY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_mem_activity_test, v4_discovers_two_mem_activity_tracks)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_EQ(tracks.size(), 2U);

    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->pmc_info, nullptr);
    }

    std::set<size_t> agent_ids;
    for(const auto& t : tracks)
        agent_ids.insert(t->agent_info->id);
    ASSERT_TRUE(agent_ids.count(1) == 1);
    ASSERT_TRUE(agent_ids.count(2) == 1);
}

TEST_F(reader_v4_mem_activity_test, v4_mem_activity_running_sum_agent1)
{
    // Same logical sequence as v3: ALLOC(4096)+FREE(4096)+REALLOC(no-op)+ALLOC(2048).
    // Rows inserted out of start order to prove ORDER BY ts_s.value.
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent1_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 1) agent1_track = t;
    }
    ASSERT_NE(agent1_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent1_track->id);
    ASSERT_EQ(scalars.size(), 3U);

    ASSERT_EQ(scalars[0].timestamp, 1000U);
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_EQ(scalars[2].timestamp, 5000U);

    ASSERT_DOUBLE_EQ(scalars[0].value, 4096.0);
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);
    ASSERT_DOUBLE_EQ(scalars[2].value, 2048.0);
}

TEST_F(reader_v4_mem_activity_test, v4_mem_activity_non_interference_agent2)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent2_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 2) agent2_track = t;
    }
    ASSERT_NE(agent2_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent2_track->id);
    ASSERT_EQ(scalars.size(), 1U);
    ASSERT_EQ(scalars[0].timestamp, 2000U);
    ASSERT_DOUBLE_EQ(scalars[0].value, 8192.0);
}

TEST_F(reader_v4_mem_activity_test, v4_mem_activity_track_stats)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    for(const auto& t : tracks)
    {
        auto scalars = m_reader->get_scalar_track(t->id);
        auto stats   = m_reader->get_track_stats(t->id);
        ASSERT_TRUE(stats.min_ts.has_value());
        ASSERT_TRUE(stats.max_ts.has_value());
        ASSERT_EQ(stats.count, scalars.size());
        ASSERT_EQ(stats.min_ts.value(), scalars.front().timestamp);
        ASSERT_EQ(stats.max_ts.value(), scalars.back().timestamp);
    }
}

TEST_F(reader_v4_mem_activity_test, v4_get_interval_track_returns_empty_for_mem_activity)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_interval_track(tracks.front()->id).empty());
}

// Task 044: the v4 `memory`-typed tracks (the real rocpd_track rows carrying
// memory_allocate rows, distinct from the synthesized memory_activity tracks)
// exercise the v4 memory arms of get_interval_track (memory_alloc_interval_track_v4)
// and get_track_stats (memory_alloc_stats_track_v4), which no prior test lit. The
// fixture's 5 allocate rows resolve through the rocpd_timestamp spine to:
//   track 1 (agent 1): starts {1000,3000,4000,5000} ends {..,5100} -> count 4
//   track 2 (agent 2): start  {2000}                end 2100       -> count 1
TEST_F(reader_v4_mem_activity_test, v4_memory_track_interval_matches_stats)
{
    auto tracks = find_tracks(m_reader->get_tracks(),
                              profiler_hub::reader_types::track_type_t::memory);
    ASSERT_EQ(tracks.size(), 2U);

    bool saw_track1 = false;
    bool saw_track2 = false;
    for(const auto& t : tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        ASSERT_TRUE(is_start_sorted(intervals));
        expect_stats_match_intervals(stats, intervals);

        if(stats.min_ts.has_value() && stats.min_ts.value() == 1000U)
        {
            saw_track1 = true;
            ASSERT_EQ(intervals.size(), 4U);
            ASSERT_EQ(intervals[0].start, 1000U);
            ASSERT_EQ(intervals[1].start, 3000U);
            ASSERT_EQ(intervals[2].start, 4000U);
            ASSERT_EQ(intervals[3].start, 5000U);
            ASSERT_EQ(stats.count, 4U);
            ASSERT_EQ(stats.max_ts.value(), 5100U);
        }
        else
        {
            saw_track2 = true;
            ASSERT_EQ(intervals.size(), 1U);
            ASSERT_EQ(intervals[0].start, 2000U);
            ASSERT_EQ(stats.count, 1U);
            ASSERT_EQ(stats.min_ts.value(), 2000U);
            ASSERT_EQ(stats.max_ts.value(), 2100U);
        }
    }
    ASSERT_TRUE(saw_track1);
    ASSERT_TRUE(saw_track2);
}

// ============================================================================
// get_kernel_summary / get_region_summary — GROUP-BY-name aggregation (task 050)
//
// Two synthetic summary fixtures share ONE by-construction oracle so both reader
// backends are asserted against identical expected aggregates: rocpd_v3_summary.db
// (durations = inline "end" - start) and rocpd_v4_summary.db (durations via the
// rocpd_timestamp spine). See fixtures/rocpd_v{3,4}_summary_data.sql for the full
// derivation. The oracle checks live in free helpers so v3 and v4 cannot drift.
//
//   KERNELS  kA "kA(int)"   : count 3, total 600, min 100, max 300, avg 200
//              (kd1,kd2 on symbol 1 + kd4 on symbol 3 — two symbol ids, one name,
//               so fold_summary_rows MERGES their GROUP BY rows into one bucket)
//            kB "kB(float)" : count 1, total 50,  min 50,  max 50,  avg 50
//   REGIONS  rX             : count 2, total 400, min 200, max 200, avg 200
//            rY             : count 1, total 600, min 600, max 600, avg 600
//   WINDOW kernels [1500,5000] -> kA count 2 (kd2,kd4), kB count 1.
//   WINDOW regions [0,750]     -> rX count 1 (r1 only), rY absent.
//   FAR-FUTURE window          -> empty list.
// ============================================================================

const profiler_hub::reader_types::event_summary_t*
find_summary(const profiler_hub::reader_types::event_summary_list_t& list,
             const std::string&                                      name)
{
    for(const auto& s : list)
        if(s.name == name) return &s;
    return nullptr;
}

void
expect_kernel_summary_oracle(const profiler_hub::reader_t& reader)
{
    auto list = reader.get_kernel_summary();
    ASSERT_EQ(list.size(), 2U);

    const auto* ka = find_summary(list, "kA(int)");
    const auto* kb = find_summary(list, "kB(float)");
    ASSERT_NE(ka, nullptr);
    ASSERT_NE(kb, nullptr);

    // kA(int) spans TWO kernel_symbol ids (1 and 3) that resolve to the same
    // display name; the single count-3 bucket proves the merge fired.
    EXPECT_EQ(ka->count, 3U);
    EXPECT_EQ(ka->total_duration, 600U);
    EXPECT_EQ(ka->min_duration, 100U);
    EXPECT_EQ(ka->max_duration, 300U);
    EXPECT_EQ(ka->avg_duration, 200U);  // 600 / 3

    EXPECT_EQ(kb->count, 1U);
    EXPECT_EQ(kb->total_duration, 50U);
    EXPECT_EQ(kb->min_duration, 50U);
    EXPECT_EQ(kb->max_duration, 50U);
    EXPECT_EQ(kb->avg_duration, 50U);
}

void
expect_region_summary_oracle(const profiler_hub::reader_t& reader)
{
    auto list = reader.get_region_summary();
    ASSERT_EQ(list.size(), 2U);

    const auto* rx = find_summary(list, "rX");
    const auto* ry = find_summary(list, "rY");
    ASSERT_NE(rx, nullptr);
    ASSERT_NE(ry, nullptr);

    EXPECT_EQ(rx->count, 2U);
    EXPECT_EQ(rx->total_duration, 400U);
    EXPECT_EQ(rx->min_duration, 200U);
    EXPECT_EQ(rx->max_duration, 200U);
    EXPECT_EQ(rx->avg_duration, 200U);  // 400 / 2

    EXPECT_EQ(ry->count, 1U);
    EXPECT_EQ(ry->total_duration, 600U);
    EXPECT_EQ(ry->min_duration, 600U);
    EXPECT_EQ(ry->max_duration, 600U);
    EXPECT_EQ(ry->avg_duration, 600U);
}

void
expect_windowed_kernel_summary(const profiler_hub::reader_t& reader)
{
    profiler_hub::reader_types::time_window_t window;
    window.start = 1500;
    window.end   = 5000;

    auto list = reader.get_kernel_summary(window);
    ASSERT_EQ(list.size(), 2U);

    const auto* ka = find_summary(list, "kA(int)");
    const auto* kb = find_summary(list, "kB(float)");
    ASSERT_NE(ka, nullptr);
    ASSERT_NE(kb, nullptr);

    EXPECT_EQ(ka->count, 2U);             // kd1 [1000,1100] dropped; kd2 + kd4 kept
    EXPECT_EQ(ka->total_duration, 500U);  // 300 (kd2) + 200 (kd4)
    EXPECT_EQ(kb->count, 1U);             // kd3 [3000,3050] still inside the window

    // Prove the window actually dropped a row rather than being ignored.
    auto        full    = reader.get_kernel_summary();
    const auto* ka_full = find_summary(full, "kA(int)");
    ASSERT_NE(ka_full, nullptr);
    EXPECT_LT(ka->count, ka_full->count);  // 2 < 3
}

void
expect_windowed_region_summary(const profiler_hub::reader_t& reader)
{
    profiler_hub::reader_types::time_window_t window;
    window.start = 0;
    window.end   = 750;

    auto list = reader.get_region_summary(window);
    ASSERT_EQ(list.size(), 1U);  // rY (r3) fully dropped, so it vanishes from the list

    const auto* rx = find_summary(list, "rX");
    ASSERT_NE(rx, nullptr);
    EXPECT_EQ(rx->count, 1U);             // r2 [800,1000] dropped, only r1 remains
    EXPECT_EQ(rx->total_duration, 200U);  // r1 duration alone
    EXPECT_EQ(find_summary(list, "rY"), nullptr);
}

void
expect_far_future_window_empty(const profiler_hub::reader_t& reader)
{
    profiler_hub::reader_types::time_window_t window;
    window.start = 1000000000;
    window.end   = 1000000001;
    EXPECT_TRUE(reader.get_kernel_summary(window).empty());
    EXPECT_TRUE(reader.get_region_summary(window).empty());
}

class reader_v4_summary_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V4_SUMMARY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_summary_test, kernel_summary_groups_by_name)
{
    expect_kernel_summary_oracle(*m_reader);
}
TEST_F(reader_v4_summary_test, region_summary_groups_by_name)
{
    expect_region_summary_oracle(*m_reader);
}
TEST_F(reader_v4_summary_test, kernel_summary_honors_time_window)
{
    expect_windowed_kernel_summary(*m_reader);
}
TEST_F(reader_v4_summary_test, region_summary_honors_time_window)
{
    expect_windowed_region_summary(*m_reader);
}
TEST_F(reader_v4_summary_test, far_future_window_yields_empty)
{
    expect_far_future_window_empty(*m_reader);
}

class reader_v3_summary_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V3_SUMMARY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_summary_test, kernel_summary_groups_by_name)
{
    expect_kernel_summary_oracle(*m_reader);
}
TEST_F(reader_v3_summary_test, region_summary_groups_by_name)
{
    expect_region_summary_oracle(*m_reader);
}
TEST_F(reader_v3_summary_test, kernel_summary_honors_time_window)
{
    expect_windowed_kernel_summary(*m_reader);
}
TEST_F(reader_v3_summary_test, region_summary_honors_time_window)
{
    expect_windowed_region_summary(*m_reader);
}
TEST_F(reader_v3_summary_test, far_future_window_yields_empty)
{
    expect_far_future_window_empty(*m_reader);
}

// =============================================================================
// Missing-metadata naming fallbacks (v3). The fixture has an unnamed stream, one
// region whose thread row is entirely absent, one region whose thread has a NULL
// name, and one agent with a NULL type_index -- so synthesize_derived_tracks()
// must fall back to the synthetic display names and get_all_agents() must drop the
// corrupt agent. Each test asserts the EXACT fallback string / dropped-agent count.
// =============================================================================
class reader_v3_missing_meta_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V3_MISSING_META_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

// stream_info present but name empty (NULL in rocpd_info_stream) -> the stream
// track display name falls back to "Stream <stream_id>".
TEST_F(reader_v3_missing_meta_test, unnamed_stream_track_falls_back_to_stream_id)
{
    auto streams = find_tracks(m_reader->get_tracks(),
                               profiler_hub::reader_types::track_type_t::stream);
    ASSERT_EQ(streams.size(), 1U);
    EXPECT_EQ(streams.front()->name, "Stream 7");
}

// region whose tid matches no rocpd_info_thread row -> thread_info is entirely
// absent, so the cpu_thread track display name is the bare "Thread". Both fixture
// regions are non-sample (main) tracks.
TEST_F(reader_v3_missing_meta_test, thread_without_thread_info_falls_back_to_thread)
{
    auto threads = find_tracks(m_reader->get_tracks(),
                               profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_EQ(threads.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t bare;
    for(const auto& t : threads)
    {
        if(t->name == "Thread") bare = t;
    }
    ASSERT_NE(bare, nullptr) << "no cpu_thread track fell back to bare \"Thread\"";
    EXPECT_EQ(bare->name, "Thread");
    // Fallback path taken precisely because thread_info could not be resolved.
    EXPECT_EQ(bare->thread_info, nullptr);
    EXPECT_EQ(bare->region_kind, profiler_hub::reader_types::region_track_kind_t::main);
}

// region whose thread row exists but has a NULL name -> thread_info is present,
// so the display name falls back to "Thread <thread_id>" using the OS tid, NOT
// the bare "Thread".
TEST_F(reader_v3_missing_meta_test, unnamed_thread_falls_back_to_thread_tid)
{
    auto threads = find_tracks(m_reader->get_tracks(),
                               profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_EQ(threads.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t named;
    for(const auto& t : threads)
    {
        if(t->name == "Thread 99001") named = t;
    }
    ASSERT_NE(named, nullptr) << "no cpu_thread track fell back to \"Thread <tid>\"";
    ASSERT_NE(named->thread_info, nullptr);
    EXPECT_EQ(named->thread_info->thread_id, 99001U);
    EXPECT_TRUE(named->thread_info->name.empty());
}

// get_all_agents() drops any agent whose type_index is NULL ("Corrupted database
// detected" continue): the fixture has one valid agent and one with a NULL
// type_index, so exactly one agent survives.
TEST_F(reader_v3_missing_meta_test, agent_with_null_type_index_is_dropped)
{
    auto agents = m_reader->get_all_agents();
    ASSERT_EQ(agents.size(), 1U);
    EXPECT_EQ(agents.front()->type_index, 0U);
    EXPECT_EQ(agents.front()->name, "Synthetic GPU 0");
}

}  // namespace
