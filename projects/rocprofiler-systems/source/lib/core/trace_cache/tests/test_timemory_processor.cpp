// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for timemory_processor — validates JSON + TXT output in timemory schema.

#include "core/trace_cache/timemory_processor.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace
{
using rocprofsys::trace_cache::backtrace_region_sample;
using rocprofsys::trace_cache::timemory_processor_t;

backtrace_region_sample
make_brs(uint64_t thread_id, const std::string& name, uint64_t beg_ns, uint64_t end_ns,
         int depth, const std::string& category = "timer_sampling",
         const std::string& track = "Thread 0 Timer (S) 12345")
{
    nlohmann::json ext;
    ext["depth"] = depth;
    return backtrace_region_sample{ 0,      thread_id, track, name, beg_ns,
                                    end_ns, category,  "{}",  "{}", ext.dump() };
}

class timemory_processor_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_tmpdir = std::filesystem::temp_directory_path() / "timemory_proc_test";
        std::filesystem::create_directories(m_tmpdir);
    }

    void TearDown() override { std::filesystem::remove_all(m_tmpdir); }

    std::string read_file(const std::string& name) const
    {
        auto          path = m_tmpdir / (name + ".json");
        std::ifstream ifs(path);
        if(!ifs.is_open()) return {};
        std::ostringstream ss;
        ss << ifs.rdbuf();
        return ss.str();
    }

    bool file_exists(const std::string& name, const std::string& ext) const
    {
        auto path = m_tmpdir / (name + ext);
        return std::filesystem::exists(path);
    }

    std::filesystem::path m_tmpdir;
};

TEST_F(timemory_processor_test, zero_samples_produces_no_files)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.finalize_processing();

    EXPECT_FALSE(file_exists("sampling_wall_clock", ".json"));
    EXPECT_FALSE(file_exists("trip_count", ".json"));
    EXPECT_FALSE(file_exists("sampling_percent", ".json"));
}

TEST_F(timemory_processor_test, single_timer_sample_produces_wall_clock_json)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.handle(make_brs(100, "main", 0, 1'000'000'000ULL, 0));
    proc.finalize_processing();

    ASSERT_TRUE(file_exists("sampling_wall_clock", ".json"));
    auto content = read_file("sampling_wall_clock");
    ASSERT_FALSE(content.empty());

    auto json = nlohmann::json::parse(content);
    ASSERT_TRUE(json.contains("timemory"));
    ASSERT_TRUE(json["timemory"].contains("sampling_wall_clock"));

    auto& metric = json["timemory"]["sampling_wall_clock"];
    EXPECT_EQ(metric["type"], "sampling_wall_clock");
    EXPECT_EQ(metric["unit_value"], 1000000000);
    EXPECT_EQ(metric["unit_repr"], "sec");

    auto& ranks = metric["ranks"];
    ASSERT_EQ(ranks.size(), 1U);
    auto& graph = ranks[0]["graph"];
    ASSERT_EQ(graph.size(), 1U);

    auto& node = graph[0];
    EXPECT_EQ(node["depth"], 0);
    EXPECT_TRUE(node.contains("hash"));
    EXPECT_TRUE(node.contains("rolling_hash"));
    EXPECT_TRUE(node.contains("entry"));
    EXPECT_TRUE(node.contains("stats"));

    auto& entry = node["entry"];
    EXPECT_EQ(entry["laps"], 1);
    EXPECT_NEAR(entry["value"].get<double>(), 1.0, 1e-6);

    auto& stats = node["stats"];
    EXPECT_NEAR(stats["sum"].get<double>(), 1.0, 1e-6);
    EXPECT_EQ(stats["count"], 1);
}

TEST_F(timemory_processor_test, single_timer_sample_produces_txt)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.handle(make_brs(100, "main", 0, 1'000'000'000ULL, 0));
    proc.finalize_processing();

    EXPECT_TRUE(file_exists("sampling_wall_clock", ".txt"));
}

TEST_F(timemory_processor_test, trip_count_produced)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.handle(make_brs(100, "fn", 0, 1'000'000'000ULL, 0));
    proc.handle(make_brs(100, "fn", 1'000'000'000ULL, 2'000'000'000ULL, 0));
    proc.finalize_processing();

    ASSERT_TRUE(file_exists("trip_count", ".json"));
    auto content = read_file("trip_count");
    auto json    = nlohmann::json::parse(content);

    auto& graph = json["timemory"]["trip_count"]["ranks"][0]["graph"];
    ASSERT_EQ(graph.size(), 1U);
    EXPECT_EQ(graph[0]["entry"]["laps"], 2);
    EXPECT_EQ(graph[0]["entry"]["value"], 2);
}

TEST_F(timemory_processor_test, sampling_percent_produced)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.handle(make_brs(100, "fn_a", 0, 1'000'000'000ULL, 0));
    proc.handle(make_brs(100, "fn_b", 1'000'000'000ULL, 3'000'000'000ULL, 0));
    proc.finalize_processing();

    ASSERT_TRUE(file_exists("sampling_percent", ".json"));
    auto content = read_file("sampling_percent");
    auto json    = nlohmann::json::parse(content);

    auto& graph = json["timemory"]["sampling_percent"]["ranks"][0]["graph"];
    ASSERT_EQ(graph.size(), 2U);

    EXPECT_EQ(graph[0]["depth"], 0);
    EXPECT_EQ(graph[1]["depth"], 0);
}

TEST_F(timemory_processor_test, overflow_category_not_in_wall_clock)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.handle(make_brs(100, "fn", 0, 1'000'000'000ULL, 0, "overflow_sampling"));
    proc.finalize_processing();

    EXPECT_FALSE(file_exists("sampling_wall_clock", ".json"));
}

TEST_F(timemory_processor_test, record_without_depth_skipped)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    backtrace_region_sample s{
        0, 1, "track", "fn", 0, 1'000'000'000ULL, "timer_sampling", "{}", "{}", "{}"
    };
    proc.handle(s);
    proc.finalize_processing();

    EXPECT_FALSE(file_exists("sampling_wall_clock", ".json"));
}

TEST_F(timemory_processor_test, prefix_format_matches_timemory)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.handle(make_brs(100, "main", 0, 1'000'000'000ULL, 0));
    proc.handle(make_brs(100, "child", 0, 500'000'000ULL, 1));
    proc.finalize_processing();

    auto  content = read_file("sampling_wall_clock");
    auto  json    = nlohmann::json::parse(content);
    auto& graph   = json["timemory"]["sampling_wall_clock"]["ranks"][0]["graph"];

    ASSERT_GE(graph.size(), 2U);
    std::string pfx0 = graph[0]["prefix"];
    std::string pfx1 = graph[1]["prefix"];

    EXPECT_TRUE(pfx0.find("|0>>> main") != std::string::npos)
        << "prefix should start with |0>>> for thread seq 0, got: " << pfx0;
    EXPECT_TRUE(pfx1.find("|_child") != std::string::npos)
        << "depth-1 prefix should contain |_child, got: " << pfx1;
}

TEST_F(timemory_processor_test, multi_thread_distinct_seq_ids)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();
    proc.handle(make_brs(100, "fn", 0, 1'000'000'000ULL, 0, "timer_sampling",
                         "Thread 0 Timer (S) 100"));
    proc.handle(make_brs(200, "fn", 0, 1'000'000'000ULL, 0, "timer_sampling",
                         "Thread 1 Timer (S) 200"));
    proc.finalize_processing();

    auto  content = read_file("sampling_wall_clock");
    auto  json    = nlohmann::json::parse(content);
    auto& graph   = json["timemory"]["sampling_wall_clock"]["ranks"][0]["graph"];

    ASSERT_EQ(graph.size(), 2U);
    std::string pfx0 = graph[0]["prefix"];
    std::string pfx1 = graph[1]["prefix"];

    EXPECT_TRUE(pfx0.find("|0>>>") != std::string::npos);
    EXPECT_TRUE(pfx1.find("|1>>>") != std::string::npos);
}

TEST_F(timemory_processor_test, non_backtrace_samples_are_noop)
{
    timemory_processor_t proc(m_tmpdir.string());
    proc.prepare_for_processing();

    rocprofsys::trace_cache::region_sample          region_s;
    rocprofsys::trace_cache::in_time_sample         in_time_s;
    rocprofsys::trace_cache::kfd_sample             kfd_s;
    rocprofsys::trace_cache::kernel_dispatch_sample kd;
    rocprofsys::trace_cache::scratch_memory_sample  sm;
    rocprofsys::trace_cache::memory_copy_sample     mc;
    rocprofsys::trace_cache::memory_allocate_sample ma;
    rocprofsys::trace_cache::pmc_event_with_sample  pmc;

    proc.handle(region_s);
    proc.handle(in_time_s);
    proc.handle(kfd_s);
    proc.handle(kd);
    proc.handle(sm);
    proc.handle(mc);
    proc.handle(ma);
    proc.handle(pmc);
    proc.finalize_processing();

    EXPECT_FALSE(file_exists("sampling_wall_clock", ".json"));
}

}  // namespace
