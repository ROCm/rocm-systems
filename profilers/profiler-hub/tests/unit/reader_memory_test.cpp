// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_test_fixture.hpp"

#include "profiler-hub/reader.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace profiler_hub;
using namespace profiler_hub::test;

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

    ASSERT_EQ(scalars[0].timestamp, 1000U);
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_EQ(scalars[2].timestamp, 5000U);

    ASSERT_DOUBLE_EQ(scalars[0].value, 4096.0);  // ALLOC +4096
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);     // FREE -4096 (recovered)
    ASSERT_DOUBLE_EQ(scalars[2].value, 2048.0);  // ALLOC +2048
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_free_agent_recovery)
{
    // The FREE row (row 3) has agent_id=NULL in the DB; its size and agent are
    // recovered from the ALLOC at the same address (4096).
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
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_non_interference_agent2)
{
    // Agent 2 has exactly 1 ALLOC; its scalar series must not include any
    // agent-1 rows or the REALLOC no-op.
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
// memory_activity time-window straddle.
// The window filter in get_scalar_track's memory_activity branch is inclusive
// on both ends (window.start <= r.start <= window.end) and applies after
// accumulation, so kept rows still reflect prior out-of-window activity.
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

    // Boundary-inclusive: rows at 3000 and 5000 are kept; the pre-window
    // ALLOC(1000)/FREE(2000) and post-window ALLOC(6000)/FREE(7000) are dropped.
    ASSERT_EQ(scalars.size(), 3U);
    ASSERT_TRUE(is_timestamp_sorted(scalars));
    ASSERT_EQ(scalars[0].timestamp, 3000U);
    ASSERT_DOUBLE_EQ(scalars[0].value, 500.0);
    ASSERT_EQ(scalars[1].timestamp, 4000U);
    ASSERT_DOUBLE_EQ(scalars[1].value, 300.0);
    ASSERT_EQ(scalars[2].timestamp, 5000U);
    ASSERT_DOUBLE_EQ(scalars[2].value, 1000.0);

    ASSERT_LT(scalars.size(), m_reader->get_scalar_track(track->id).size());
}

TEST_F(reader_v3_mem_activity_window_test, time_window_start_only_drops_earlier_rows)
{
    auto track = mem_activity_track();
    ASSERT_NE(track, nullptr);

    // end is unset to exercise the has_value() guard on the end-filter branch.
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

// The v4 `memory`-typed tracks (real rocpd_track rows carrying memory_allocate
// data) are distinct from the synthesized memory_activity tracks and exercise
// get_interval_track/get_track_stats's v4 memory arms
// (memory_alloc_interval_track_v4 / memory_alloc_stats_track_v4).
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
