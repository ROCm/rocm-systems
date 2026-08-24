// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace profiler_hub::test
{

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

    // Seeds data so callers' detail/property getters have real data to resolve.
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

inline reader_types::event_type_t
type_of(const reader_types::event_id_t& id)
{
    return reader_types::detail::event_id_access::type(id);
}

inline uint64_t
flow_id_value(const reader_types::flow_id_t& fid)
{
    return reader_types::detail::flow_id_access::value(fid);
}

inline int
count_interval_resolutions(const reader_t& r, const reader_types::event_id_t& id)
{
    return r.get_event_info(id).has_value() ? 1 : 0;
}

inline reader_types::track_info_ptr_t
find_first_track(const reader_types::track_info_list_t& tracks,
                 reader_types::track_type_t             type)
{
    for(const auto& t : tracks)
    {
        if(t->type == type) return t;
    }
    return nullptr;
}

// Test-only: the public API treats event_id_t as opaque (equality / ordering /
// hashing only).
inline size_t
row_id_of(const reader_types::event_id_t& id)
{
    return reader_types::detail::event_id_access::row_id(id);
}

inline const reader_types::arg_value_t*
find_prop(const reader_types::event_info_t& d, const std::string& key)
{
    for(const auto& p : d.properties)
    {
        if(p.key == key) return &p.value;
    }
    return nullptr;
}

inline reader_types::track_info_list_t
find_tracks(const reader_types::track_info_list_t& tracks,
            reader_types::track_type_t             type)
{
    reader_types::track_info_list_t out;
    for(const auto& t : tracks)
    {
        if(t->type == type) out.push_back(t);
    }
    return out;
}

// Checks conformance to get_interval_track's documented ordering contract.
inline bool
is_start_sorted(const reader_types::interval_entry_list_t& v)
{
    for(size_t i = 1; i < v.size(); ++i)
    {
        if(v[i].start < v[i - 1].start) return false;
    }
    return true;
}

// Checks conformance to get_scalar_track's documented ordering contract.
inline bool
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
inline void
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
inline void
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

}  // namespace profiler_hub::test
