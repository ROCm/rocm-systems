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

}  // namespace
