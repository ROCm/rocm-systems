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

using namespace profiler_hub;
using namespace profiler_hub::test;

// Mint the handle for a SQL-seeded fixture row by its EXPLICIT id. Unlike the
// writer's 0-based autoincrementer (make_event_id, in reader_flows_test.cpp),
// synthetic SQL fixtures assign explicit 1-based ids (e.g. rocpd_sample.id = 1),
// so no ordinal->raw shift applies — the db_id passed here IS the raw id.
reader_types::event_id_t
make_sql_event_id(reader_types::event_type_t type, size_t db_id)
{
    return reader_types::detail::event_id_access::make(type, db_id);
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

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_pmc_info_populated)
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

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_interval_track_count_and_order)
{
    // Rows are inserted out of start order (kd 2 first, kd 1 second); proves ORDER BY
    // start.
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

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_interval_resolves_as_kernel_dispatch)
{
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
    auto tracks =
        find_tracks(m_reader->get_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_scalar_track(tracks.front()->id).empty());
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
    // Timestamps inserted out of value order (kd 2 rows precede kd 1 rows); proves
    // ORDER BY ts_s.value returns kd 1 before kd 2.
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

class reader_v3_amb_pmc_test : public ::testing::Test
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

    std::string                              m_database_path{ ROCPD_DB_V3_AMB_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_amb_pmc_test, v3_ambiguous_pmc_id_flagged)
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

TEST_F(reader_v3_amb_pmc_test, v3_exactly_one_ambiguous_pmc)
{
    auto   pmc_list        = m_reader->get_all_pmc_info();
    size_t ambiguous_count = 0;
    for(const auto& p : pmc_list)
    {
        if(p->ambiguous) ++ambiguous_count;
    }
    EXPECT_EQ(ambiguous_count, 1U);
}

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

// Fixture: rocpd_v4_amb_cls.db — a single rocpd_track row (id=1) referenced by both
// rocpd_sample/rocpd_pmc_event (counter set) and rocpd_memory_allocate (memory set);
// build_v4_tracks() must detect the overlap and set ambiguous_classification=true.

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

    // Fixture also yields a synthetic memory_activity track from the same
    // rocpd_memory_allocate row.
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
    EXPECT_EQ(counter_track->type, profiler_hub::reader_types::track_type_t::counter);
    EXPECT_TRUE(counter_track->ambiguous_classification);
}

TEST_F(reader_v4_amb_cls_test, v4_non_ambiguous_track_not_flagged)
{
    // The main v4 fixture has no overlapping track_ids; no track should be flagged.
    auto storage = std::make_unique<profiler_hub::storage_t>(ROCPD_DB_V4_PATH, "");
    auto reader  = std::make_shared<profiler_hub::reader_t>(std::move(storage));

    for(const auto& t : reader->get_tracks())
    {
        EXPECT_FALSE(t->ambiguous_classification)
            << "unexpected ambiguous_classification on track id=" << t->id.value;
    }
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
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    ASSERT_EQ(counter->name, "GRBM_COUNT");
    ASSERT_NE(counter->agent_info, nullptr);
    ASSERT_NE(counter->thread_info, nullptr);

    // v4.0 has one pmc per event (no event_id fan-out).
    ASSERT_NE(counter->pmc_info, nullptr);
    ASSERT_EQ(counter->pmc_info->name, "GRBM_COUNT");
    // GRBM_COUNT is pmc 1 here.
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
    auto details = m_reader->get_event_info(
        make_sql_event_id(profiler_hub::reader_types::event_type_t::sample, 1));
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->ts, 3000U);
    ASSERT_FALSE(details->te.has_value());
}

TEST_F(reader_v4_counter_test, v4_get_event_info_counter_sample_carries_name_and_value)
{
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
    auto tracks = m_reader->get_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);
    ASSERT_TRUE(m_reader->get_interval_track(counter->id).empty());
}

TEST_F(reader_v4_counter_test, v4_get_scalar_track_on_non_counter_returns_empty)
{
    auto tracks = m_reader->get_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    ASSERT_TRUE(m_reader->get_scalar_track(cpu->id).empty());
}

TEST_F(reader_v4_counter_test, v4_get_track_stats_counter_matches_scalar_slice)
{
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
    // Empty track (count 0, nullopt bounds) is the expected result, not an error.
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
    // Track 3's pmc_event references pmc_id=99, which exists in rocpd_info_pmc with an
    // empty name; the empty-name guard prevents it from overwriting the track name, so
    // display name falls back to "FallbackCounterV4".
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

    ASSERT_EQ(fallback_counter->name, "FallbackCounterV4");
    ASSERT_FALSE(fallback_counter->name.empty());
    ASSERT_NE(fallback_counter->pmc_info, nullptr);
    ASSERT_TRUE(fallback_counter->pmc_info->name.empty());
    // Non-fallback path still intact: the GRBM_COUNT track carries pmc_info.
    auto grbm =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(grbm, nullptr);
    ASSERT_EQ(grbm->name, "GRBM_COUNT");
    ASSERT_NE(grbm->pmc_info, nullptr);
}
