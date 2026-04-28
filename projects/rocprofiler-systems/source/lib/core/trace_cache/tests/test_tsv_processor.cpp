// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD tests for tsv_processor — task #28 (session-snapshot #21).
//
// tsv_processor consumes backtrace_region_sample records from the trace_cache
// and emits the same four-file TSV set as native_report_writer:
//   sampling_wall_clock.tsv / sampling_cpu_clock.tsv /
//   sampling_percent.tsv / trip_count.tsv
//
// The extdata field carries { "depth": <int> } (locked schema).
// Records without a valid "depth" in extdata are silently skipped.
//
// Tests are RED until tsv_processor is implemented.

#include "core/trace_cache/tsv_processor.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using rocprofsys::trace_cache::backtrace_region_sample;
using rocprofsys::trace_cache::tsv_processor_t;

// Build a minimal backtrace_region_sample with the locked extdata schema.
backtrace_region_sample
make_brs(uint64_t thread_id, const std::string& name, uint64_t beg_ns, uint64_t end_ns,
         int depth, const std::string& category = "timer_sampling")
{
    nlohmann::json ext;
    ext["depth"] = depth;
    return backtrace_region_sample{ 0,      thread_id, "track", name, beg_ns,
                                    end_ns, category,  "{}",    "{}", ext.dump() };
}

// Split TSV output into data rows (skip comment lines).
static std::vector<std::vector<std::string>>
parse_data_rows(const std::string& tsv)
{
    std::vector<std::vector<std::string>> result;
    std::istringstream                    ss(tsv);
    std::string                           line;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        std::string              tok;
        for(char c : line)
        {
            if(c == '\t')
            {
                cols.push_back(tok);
                tok.clear();
            }
            else
            {
                tok += c;
            }
        }
        cols.push_back(tok);
        result.push_back(std::move(cols));
    }
    return result;
}

static int
count_header_lines(const std::string& tsv)
{
    std::istringstream ss(tsv);
    std::string        line;
    int                n = 0;
    while(std::getline(ss, line))
        if(!line.empty() && line[0] == '#') ++n;
    return n;
}

static bool
has_header_key(const std::string& tsv, const std::string& key)
{
    std::istringstream ss(tsv);
    std::string        line;
    while(std::getline(ss, line))
        if(line[0] == '#' && line.find(key) != std::string::npos) return true;
    return false;
}
}  // namespace

// ── Construction and header output ──────────────────────────────────────────

TEST(tsv_processor, stream_injection_constructor_writes_headers)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.finalize_processing();

    EXPECT_GE(count_header_lines(wall.str()), 3)
        << "wall_clock.tsv must have >= 3 header comment lines";
    EXPECT_TRUE(has_header_key(wall.str(), "metric:"));
    EXPECT_TRUE(has_header_key(wall.str(), "unit:"));
    EXPECT_TRUE(has_header_key(wall.str(), "columns:"));
    EXPECT_TRUE(has_header_key(wall.str(), "depth"));
}

TEST(tsv_processor, zero_samples_produces_empty_data_rows)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.finalize_processing();

    EXPECT_TRUE(parse_data_rows(wall.str()).empty());
}

// ── Non-backtrace samples are no-ops ────────────────────────────────────────

TEST(tsv_processor, non_backtrace_sample_types_are_ignored)
{
    std::ostringstream                      wall, cpu, pct, trip;
    tsv_processor_t                         proc(wall, cpu, pct, trip);
    rocprofsys::trace_cache::region_sample  region_s;
    rocprofsys::trace_cache::in_time_sample in_time_s;
    rocprofsys::trace_cache::kfd_sample     kfd_s;
    proc.prepare_for_processing();
    proc.handle(region_s);
    proc.handle(in_time_s);
    proc.handle(kfd_s);
    proc.finalize_processing();

    EXPECT_TRUE(parse_data_rows(wall.str()).empty())
        << "non-backtrace samples must not produce data rows";
}

// ── Single backtrace_region_sample round-trip ────────────────────────────────

TEST(tsv_processor, single_sample_one_data_row_wall_clock)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(1, "main", 0, 1'000'000'000ULL, 0));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_EQ(rows.size(), 1U) << "one sample must produce exactly one data row";
    EXPECT_EQ(rows[0].size(), 11U) << "wall_clock row must have 11 columns (thread_id "
                                      "depth label count sum mean min "
                                      "max var stddev pct_self)";
}

TEST(tsv_processor, single_sample_thread_id_matches)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(42, "fn", 0, 1'000'000'000ULL, 0));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0][0], "42") << "thread_id column must match sample's thread_id";
}

TEST(tsv_processor, single_sample_depth_matches_extdata)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(1, "fn", 0, 1'000'000'000ULL, 3));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0][1], "3") << "depth column must equal extdata[\"depth\"]";
}

TEST(tsv_processor, single_sample_label_matches_name)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(1, "my_function", 0, 1'000'000'000ULL, 0));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0][2], "my_function") << "label column must equal sample.name";
}

TEST(tsv_processor, single_sample_sum_is_duration_seconds)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    // 2 seconds exactly
    proc.handle(make_brs(1, "fn", 0, 2'000'000'000ULL, 0));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0][4], "2.000000")
        << "sum must be 2.000000 s (6 decimal places) for a 2-second sample";
}

TEST(tsv_processor, single_sample_var_zero_not_nan)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(1, "fn", 0, 1'000'000'000ULL, 0));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0][8], "0.000000")
        << "var must be 0.000000 when count=1 (Bessel convention, not NaN)";
}

// ── Aggregation: two records with same (thread_id, depth, label) ─────────────

TEST(tsv_processor, two_samples_same_key_aggregated)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(1, "fn", 0, 1'000'000'000ULL, 0));
    proc.handle(make_brs(1, "fn", 1'000'000'000ULL, 2'000'000'000ULL, 0));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_EQ(rows.size(), 1U) << "two samples with same key must aggregate into one row";
    EXPECT_EQ(rows[0][3], "2") << "count must be 2";
    EXPECT_EQ(rows[0][4], "2.000000") << "sum must be 2.000000 s";
}

// ── Extdata without "depth" is silently skipped ──────────────────────────────

TEST(tsv_processor, record_without_depth_in_extdata_is_skipped)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    // extdata = "{}" — no "depth" key
    backtrace_region_sample s{
        0, 1, "track", "fn", 0, 1'000'000'000ULL, "timer_sampling", "{}", "{}", "{}"
    };
    proc.handle(s);
    proc.finalize_processing();

    EXPECT_TRUE(parse_data_rows(wall.str()).empty())
        << "backtrace_region_sample without extdata[\"depth\"] must be silently skipped";
}

// ── Only backtrace_region_sample with timer_sampling category goes to wall ───

TEST(tsv_processor, overflow_category_not_in_wall_clock_file)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(1, "fn", 0, 1'000'000'000ULL, 0, "overflow_sampling"));
    proc.finalize_processing();

    // overflow records must NOT appear in the wall_clock file
    auto wall_rows = parse_data_rows(wall.str());
    EXPECT_TRUE(wall_rows.empty())
        << "overflow_sampling records must not appear in wall_clock.tsv";
}

// ── trip_count.tsv ──────────────────────────────────────────────────────────

TEST(tsv_processor, trip_count_produced_for_timer_sample)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(1, "fn", 0, 1'000'000'000ULL, 2));
    proc.finalize_processing();

    auto trip_rows = parse_data_rows(trip.str());
    ASSERT_EQ(trip_rows.size(), 1U);
    EXPECT_EQ(trip_rows[0][0], "1") << "thread_id";
    EXPECT_EQ(trip_rows[0][1], "2") << "depth from extdata";
    EXPECT_EQ(trip_rows[0][2], "fn") << "label";
    EXPECT_EQ(trip_rows[0][3], "1") << "count";
}

// ── Multi-thread: rows distinguished by thread_id ───────────────────────────

TEST(tsv_processor, multi_thread_rows_present_for_each_tid)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();
    proc.handle(make_brs(10, "fn", 0, 1'000'000'000ULL, 0));
    proc.handle(make_brs(20, "fn", 0, 1'000'000'000ULL, 0));
    proc.finalize_processing();

    auto rows = parse_data_rows(wall.str());
    ASSERT_EQ(rows.size(), 2U) << "two threads must produce two rows";
    bool has_10 = false, has_20 = false;
    for(auto& r : rows)
    {
        if(r[0] == "10") has_10 = true;
        if(r[0] == "20") has_20 = true;
    }
    EXPECT_TRUE(has_10);
    EXPECT_TRUE(has_20);
}

// ── processor_t concept: non-AMD-SMI sample types must compile and be no-ops ──
// Note: gpu_pmc_sample, ainic_pmc_sample, cpu_pmc_sample require AMD-SMI headers
// and are not tested here. Their no-op overloads are provided in
// tsv_processor_adapter.hpp (production-only).

TEST(tsv_processor, handle_non_amdsmi_sample_types_compile_and_noop)
{
    std::ostringstream wall, cpu, pct, trip;
    tsv_processor_t    proc(wall, cpu, pct, trip);
    proc.prepare_for_processing();

    rocprofsys::trace_cache::kernel_dispatch_sample kd;
    rocprofsys::trace_cache::scratch_memory_sample  sm;
    rocprofsys::trace_cache::memory_copy_sample     mc;
    rocprofsys::trace_cache::memory_allocate_sample ma;
    rocprofsys::trace_cache::pmc_event_with_sample  pmc;

    proc.handle(kd);
    proc.handle(sm);
    proc.handle(mc);
    proc.handle(ma);
    proc.handle(pmc);
    proc.finalize_processing();

    EXPECT_TRUE(parse_data_rows(wall.str()).empty());
}
