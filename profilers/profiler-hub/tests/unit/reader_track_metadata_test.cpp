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
#include <set>
#include <string>
#include <utility>

using namespace profiler_hub;
using namespace profiler_hub::test;

// =============================================================================
// v3 track-type x schema switch-arm coverage: rocpd_v3_track_shapes.db carries one
// track of each v3 dma/memory/cpu_thread shape that other v3 fixtures leave untested
// in get_track_stats/get_interval_track -- dma queue-only/agent-only/queue+agent,
// memory queue+agent/queue-only/neither, and a cpu_thread SAMPLE track.
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

// v3 dma-by-destination-agent fixture: the crossed 2-agent x 2-stream x 12 = 48
// memory_copy pattern (fixtures/rocpd_v3_dma_agent_data.sql) that proves dma tracks
// partition by dst_agent_id, not stream_id -- the reproducible in-tree stand-in for
// roc-optiq's rocpd-transpose.db.
class reader_v3_dma_agent_test : public ::testing::Test
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

    std::string m_database_path{ ROCPD_DB_V3_DMA_AGENT_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_dma_agent_test, dma_tracks_partition_by_destination_agent)
{
    // The 48 memory copies fully cross two destination agents (id 1, 2) with two
    // streams (12 events per agent/stream cell), all on one queue. Keyed by
    // (nid,pid,queue_id,dst_agent_id) this MUST yield exactly 2 dma tracks -- one per
    // destination agent, 24 events each -- matching Optiq's
    // GetRocprofMemoryCopyTrackQuery by-agent swimlane grouping. Guards against a
    // regression to by-stream keying, which would yield 2 tracks of 24 each spanning
    // both agents (the exact inverse).
    auto tracks = m_reader->get_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 2U);

    std::set<size_t> track_agent_ids;
    for(const auto& track : dma)
    {
        ASSERT_NE(track->agent_info, nullptr)
            << "dma track must resolve agent_info from dst_agent_id";
        ASSERT_EQ(track->stream_info, nullptr)
            << "dma track must not carry stream_info under the by-agent key";
        track_agent_ids.insert(track->agent_info->id);

        auto intervals = m_reader->get_interval_track(track->id);
        ASSERT_EQ(intervals.size(), 24U)
            << "each destination-agent track holds 24 copies (12 per stream)";

        std::set<std::string> stream_names;
        for(const auto& ev : intervals)
        {
            auto details = m_reader->get_event_info(ev.id);
            ASSERT_TRUE(details.has_value());
            auto* dst_agent_id = find_prop(*details, "dst_agent_id");
            ASSERT_NE(dst_agent_id, nullptr);
            ASSERT_TRUE(std::holds_alternative<uint64_t>(*dst_agent_id));
            ASSERT_EQ(std::get<uint64_t>(*dst_agent_id), track->agent_info->id)
                << "copy in a dma track must target that track's destination agent";
            stream_names.insert(ev.display_name);
        }
        ASSERT_EQ(stream_names.size(), 2U) << "a destination-agent track must span both "
                                              "streams (by-agent, not by-stream)";
        ASSERT_TRUE(stream_names.count("copyStreamX") == 1);
        ASSERT_TRUE(stream_names.count("copyStreamY") == 1);
    }

    ASSERT_EQ(track_agent_ids.size(), 2U);
    ASSERT_TRUE(track_agent_ids.count(1) == 1);
    ASSERT_TRUE(track_agent_ids.count(2) == 1);
}

// =============================================================================
// Missing-metadata naming fallbacks (v3). The fixture has an unnamed stream, one
// region whose thread row is entirely absent, one region whose thread has a NULL
// name, and one agent with a NULL type_index -- so synthesize_derived_tracks()
// must fall back to the synthetic display names and get_all_agents() must drop the
// corrupt agent.
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

    std::string m_database_path{ ROCPD_DB_V3_MISSING_META_PATH };
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
// absent, so the cpu_thread track display name is the bare "Thread".
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
// detected" continue).
TEST_F(reader_v3_missing_meta_test, agent_with_null_type_index_is_dropped)
{
    auto agents = m_reader->get_all_agents();
    ASSERT_EQ(agents.size(), 1U);
    EXPECT_EQ(agents.front()->type_index, 0U);
    EXPECT_EQ(agents.front()->name, "Synthetic GPU 0");
}

// A v3 counter track whose metric name contains " [" ("TCC_HIT [sum] [0]") defeats
// ranked_pmc_resolver's ordinal strip (cuts at the first " [", yielding "TCC_HIT",
// matching no pmc name); the name-match rank collapses and the pmc_id tiebreaker
// alone selects the pmc. This fixture places the correct pmc at the lower id, so
// resolution is correct only by coincidence -- a latent, non-fatal degradation.
class reader_v3_bracket_name_test : public ::testing::Test
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

    std::string m_database_path{ ROCPD_DB_V3_BRACKET_NAME_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_bracket_name_test, delimiter_in_metric_name_resolves_via_pmc_id_tiebreak)
{
    auto tracks = m_reader->get_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    // Co-sampled with a sibling pmc under one event.
    ASSERT_EQ(counters.size(), 1U);

    const auto& counter = counters.front();
    ASSERT_NE(counter->pmc_info, nullptr);

    EXPECT_EQ(counter->pmc_info->pmc_id, 1U);
    EXPECT_EQ(counter->pmc_info->name, "TCC_HIT [sum]");
    // Q9.
    EXPECT_EQ(counter->name, "TCC_HIT [sum]");
}
