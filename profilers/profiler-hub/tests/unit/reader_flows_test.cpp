// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_test_fixture.hpp"

#include "profiler-hub/reader.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace profiler_hub;
using namespace profiler_hub::test;

// ordinal is 1-based; the writer's per-type autoincrementer is 0-based
// (source/autoincrementer.hpp), so raw id = ordinal - 1.
reader_types::event_id_t
make_event_id(reader_types::event_type_t type, size_t ordinal)
{
    return reader_types::detail::event_id_access::make(type, ordinal - 1);
}

// trace_environment.track_name is left unset here and in the seed_* helpers below,
// so no rocpd_sample is minted — regions collapse onto a single cpu_thread/main track.

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
seed_region(writer_t&                 writer,
            size_t                    stack_id,
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
seed_kernel_dispatch(writer_t&                 writer,
                     size_t                    stack_id,
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
seed_memory_copy(writer_t&                 writer,
                 size_t                    stack_id,
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
seed_memory_alloc(writer_t&                 writer,
                  size_t                    stack_id,
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
// Clique fixture: stack 1000 = one of each type (region1/kd1/mc1/ma1); stacks
// 2000/3000/4000/5000 = same-type sibling pairs (region2/3, kd2/3, mc2/3, ma2/3);
// stack 0 = region4 (excluded). 11 undirected clique pairs -> 7 directed edges.
// ---------------------------------------------------------------------------
class reader_v3_clique_test : public reader_test
{
protected:
    void SetUp() override
    {
        reader_test::SetUp();
        auto writer = make_writer();
        seed_flow_identity(*writer);

        seed_region(*writer, 1000, 1000, 1100);  // region 1
        seed_region(*writer, 2000, 2000, 2100);  // region 2
        seed_region(*writer, 2000, 2050, 2150);  // region 3
        seed_region(*writer, 0, 9000, 9100);     // region 4 (stack 0 -> excluded)

        seed_kernel_dispatch(*writer, 1000, 1200, 1300);  // kd 1
        seed_kernel_dispatch(*writer, 3000, 3000, 3100);  // kd 2
        seed_kernel_dispatch(*writer, 3000, 3050, 3150);  // kd 3

        seed_memory_copy(*writer, 1000, 1400, 1500);  // mc 1
        seed_memory_copy(*writer, 4000, 4000, 4100);  // mc 2
        seed_memory_copy(*writer, 4000, 4050, 4150);  // mc 3

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
    // 11 undirected pairs -> 7 directed edges: the 3 cross-type region->gpu legs were
    // already single-direction; the 4 same-type sibling sets each de-dup from two
    // ordered pairs to one.
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
    // memory_allocate 1 share raw row id 1 across different per-type tables; they must
    // mint to four distinct handles.
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

    auto none = m_reader->get_flows_for_chain(
        profiler_hub::reader_types::detail::flow_id_access::make(99));
    EXPECT_TRUE(none.empty());

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

    auto none = m_reader->get_flows_for_event(make_event_id(et::region, 999));
    EXPECT_TRUE(none.empty());
}

TEST_F(reader_v3_clique_test,
       get_flows_in_window_empty_window_and_tracks_equals_get_flows)
{
    // Empty window + empty tracks + max_edges 0 is a pure pass-through of get_flows({})
    // — same edges, same source/dest/flow_id/kind, no cap.
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
    // Window overlap uses the edge extent, boundary-inclusive. [2000,5150] captures
    // the four same-type sibling edges (extents start >= 2000); region1's three legs
    // (ehi <= 1700 < 2000) fall out.
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

    profiler_hub::reader_types::time_window_t outside;
    outside.start = 6000;
    EXPECT_TRUE(m_reader->get_flows_in_window({}, outside, 0).empty());
}

TEST_F(reader_v3_clique_test, get_flows_in_window_filters_by_track_membership)
{
    // An edge is kept iff AT LEAST ONE endpoint sits on a listed track. The single
    // cpu_thread track carries regions 1/2/3, so scoping to it keeps region1's three
    // legs (source-only membership) plus region2->region3 (both endpoints), and drops
    // the kd/mc/ma sibling edges (neither endpoint on a region track).
    auto cpu = find_first_track(m_reader->get_tracks(),
                                profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    EXPECT_EQ(m_reader->get_flows_in_window({ cpu->id }, {}, 0).size(), 4U);

    EXPECT_EQ(m_reader->get_flows_in_window({}, {}, 0).size(), 7U);
}

TEST_F(reader_v3_clique_test, get_flows_in_window_decimates_by_latency_stably)
{
    // Cap to max_edges by descending arrow-span latency. region1's three legs have the
    // only nonzero latencies (500 > 300 > 100), so max_edges 3 returns exactly those
    // three; the four zero-latency sibling edges are dropped.
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
// Flow-ordering / tie-break fixture: crafts the two degenerate-but-legitimate
// shapes the clique fixture never hits — an equal-start pair (direction tie-break)
// and two same-source legs at identical clamped-zero latency (windowed decimation
// dest tie-break). parent_stack_id NULL throughout.
// ---------------------------------------------------------------------------
class reader_v3_flow_order_test : public reader_test
{
protected:
    void SetUp() override
    {
        reader_test::SetUp();
        auto writer = make_writer();
        seed_flow_identity(*writer);

        seed_region(*writer, 1000, 5000, 5100);  // region 1 (equal-start pair)
        seed_region(*writer, 1000, 5000, 5100);  // region 2 (equal-start pair)
        seed_region(*writer, 2000, 6000, 6500);  // region 3 (encloses its children)

        seed_kernel_dispatch(*writer, 2000, 6100, 6200);  // kd 1 (inside region 3)
        seed_kernel_dispatch(*writer, 3000, 7000, 7100);  // kd 2 (equal-start pair)
        seed_kernel_dispatch(*writer, 3000, 7000, 7100);  // kd 3 (equal-start pair)

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
    // Stack 1000: region 1 and region 2 both start at 5000 with parent_stack_id NULL,
    // so neither the lineage nor start-ts branch can decide; direction falls to the
    // tie-break src = lower event_id_t handle. region 1's handle is lower, so the edge
    // must be region 1 -> region 2 (never the reverse).
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
    // Stack 2000: region 3 [6000,6500] sources kd1 (6100) and mc1 (6200); both start
    // before region 3 ends, so both arrow-span latencies clamp to 0 — equal latency AND
    // equal source, forcing the decimation tie-break `a.dest < b.dest`. kd1's handle <
    // mc1's, so the survivor must be region 3 -> kd1.
    profiler_hub::reader_types::time_window_t win;
    win.start = 6000;
    win.end   = 6600;  // excludes the stack-1000 (5000) and stack-3000 (7000) edges

    auto both = m_reader->get_flows_in_window({}, win, 0);
    ASSERT_EQ(both.size(), 2U);
    for(const auto& f : both)
        EXPECT_EQ(f.source, make_event_id(et::region, 3));

    auto top1 = m_reader->get_flows_in_window({}, win, 1);
    ASSERT_EQ(top1.size(), 1U);
    EXPECT_EQ(top1.front().source, make_event_id(et::region, 3));
    EXPECT_EQ(top1.front().dest, make_event_id(et::kernel_dispatch, 1));

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
// Flat-clique edge fixture: the single flow test in the edge fixture family. Each
// non-zero stack is one region + one GPU event (region source, one GPU-type dest,
// no siblings); stack 0 is a lone excluded region.
// ---------------------------------------------------------------------------
class reader_v3_edge_flow_test : public reader_test
{
protected:
    void SetUp() override
    {
        reader_test::SetUp();
        auto writer = make_writer();
        seed_flow_identity(*writer);

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
        ASSERT_GT(flow_id_value(f.flow_id), 0U);
        ASSERT_TRUE(f.kind == fk::launch_to_dispatch ||
                    f.kind == fk::copy_submit_to_exec);
    }
}

// Writer-seeded reproduction of the edge oracle's non-counter matrix plus its
// two discoverable counters.
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
        seed_named_region(*writer, "RegionGamma", 0, 3000, 3500);    // region id 0
        seed_named_region(*writer, "RegionBeta", 200, 2000, 2500);   // region id 1
        seed_named_region(*writer, "RegionAlpha", 100, 1000, 5000);  // region id 2
        seed_named_region(*writer, "RegionDelta", 400, 6000, 6500);  // region id 3

        seed_kd(*writer, /*queue*/ 1, /*stream*/ 1, 1200, 1300);
        seed_kd(*writer, /*queue*/ 2, /*stream*/ 1, 1400, 1500);
        seed_kd(*writer, /*queue*/ 1, /*stream*/ 1, 1600, 1700);

        seed_mc(*writer, /*stream*/ 1, 2100, 2150);
        seed_mc(*writer, /*stream*/ 1, 2200, 2250);
        seed_mc(*writer, /*stream*/ 2, 2400, 2450);

        // 1 memory allocate on agent {GPU,0} (id 1), queue NULL, stream 1 -> one
        // memory track + one memory_activity track; also the 6th stream-1 event.
        seed_ma(*writer, /*stream*/ 1, 6100, 6200);

        // 2 discoverable counters. Samples inserted out of timestamp order so the
        // reader's ORDER BY timestamp is exercised.
        seed_counter(*writer,
                     "GRBM_COUNT",
                     "grbm_track",
                     { { 3000, 30.5 }, { 1000, 10.5 }, { 2000, 20.5 } });
        seed_counter(
            *writer, "SQ_WAVES", "sq_waves_track", { { 500, 5.0 }, { 1500, 15.0 } });

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
        // primary key; stream_id 1 must carry the 6-event stream). It carries no events,
        // so it never surfaces as a synthesized stream track.
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

    // Registers a PMC + track, then inserts one PMC-backed sample per (ts, value).
    // pmc/track names are string_view — callers must pass string literals (static
    // storage).
    void seed_counter(
        writer_t&                                                        writer,
        std::string_view                                                 pmc_name,
        std::string_view                                                 track_name,
        const std::vector<std::pair<reader_types::timestamp_t, double>>& samples) const
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
    // Stats must equal MIN/MAX/COUNT over the exact interval/scalar slice for every
    // track — covering cpu_thread, gpu_queue, dma (the queue_id+dst_agent_id NULL
    // "neither" variant; the "qa" variant is covered by the dma-by-agent fixture),
    // and counter in one pass.
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
    // The only fixture exercising all three UNION legs of a stream track, including
    // memory_allocate. Oracle: stream 1 (1,1,1) = 3 kernel_dispatch + 2 memory_copy +
    // 1 memory_allocate = 6 events, ORDER BY start: kd3(1200) kd2(1400) kd1(1600)
    // mc3(2100) mc1(2200) ma1(6100). Stream 2 (1,1,2) = 1 memory_copy (mc2, 2400).
    // op_kind is retired: each handle resolves through exactly one get_*_details()
    // accessor, which is what's asserted here.
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
    ASSERT_NE(t->agent_info, nullptr);
    EXPECT_EQ(t->agent_info->id, 1U);
    EXPECT_EQ(t->queue_info, nullptr);

    auto intervals = m_reader->get_interval_track(t->id);
    ASSERT_EQ(intervals.size(), 1U);
    EXPECT_EQ(intervals.front().start, 6100U);
    EXPECT_EQ(intervals.front().end, 6200U);

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
    auto tracks  = m_reader->get_tracks();
    auto mem_trk = find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory);
    ASSERT_EQ(mem_trk.size(), 1U);

    auto intervals = m_reader->get_interval_track(mem_trk.front()->id);
    expect_stats_match_intervals(m_reader->get_track_stats(mem_trk.front()->id),
                                 intervals);
}

// ===========================================================================
// SQL-seeded edge tests: load the committed SQL fixture (fixtures/
// rocpd_v3_edge_data.sql -> rocpd_v3_edge.db, built at configure time) because
// the writer cannot yet produce these shapes: a bare non-pmc rocpd_sample track
// (a correct writer must never emit one), rocpd_arg rows on kernel_dispatch/
// memory_copy/memory_allocate events (the writer only carries args on regions),
// and counter/track identity with NULL pid/tid or an empty-pmc-name fallback,
// authored directly on rocpd_track.
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
    // Synthesis adds 1 memory_activity (1 alloc row, agent_id=1) => total 12.
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
    ASSERT_EQ(with_thread, 1);
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

    ASSERT_EQ(no_tid_counter->thread_info, nullptr);
    ASSERT_EQ(no_tid_counter->agent_info, nullptr);

    // Counter WITH tid -> thread_info populated; a case real capture DBs don't
    // exercise, hence the synthetic fixture.
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

    ASSERT_EQ(fallback_counter->name, "FallbackCounter");
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
    auto                                           tracks      = m_reader->get_tracks();
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
    auto                                           tracks      = m_reader->get_tracks();
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
