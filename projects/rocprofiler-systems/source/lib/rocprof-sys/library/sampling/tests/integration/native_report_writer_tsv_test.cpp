// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD tests for native_report_writer TSV format — Phase H.
//
// These tests are RED until code-writer implements the new native_report_writer
// per the L4 TSV spec in requirements.md.
//
// L4 format decisions:
//   - wall_clock.tsv / cpu_clock.tsv: 11 columns (thread_id depth label count
//     sum mean min max var stddev pct_self)
//   - percent.tsv: 4 columns (thread_id label count sum), flat_scope (depth=0, dedup)
//   - trip_count.tsv: 4 columns (thread_id depth label count)
//   - Each file has 3 header comment lines: "# metric:", "# unit:", "# columns:"
//   - VAR=0 when n<2 (not NaN)
//   - Labels untruncated
//   - 6 decimal places for wall/cpu; 3 for percent
//   - Tab delimiter only, no padding
//
// Tests use ostream injection: native_report_writer(wall_out, cpu_out, pct_out, trip_out)
// where each is std::ostringstream. hw_counters stream omitted (PAPI-gated).
//
// Criteria: NFR-T-7, NFR-P-4, L4, AC-10 (output from post_process).

#include <gtest/gtest.h>

#include "sampling/data/stack_frame.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/src/native_report_writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using namespace rocprofsys::sampling;

// ── TSV parsing helpers ───────────────────────────────────────────────────────

struct tsv_row
{
    std::string thread_id;
    std::string depth;
    std::string label;
    std::string count;
    std::string sum;
    std::string mean;
    std::string min_val;
    std::string max_val;
    std::string var;
    std::string stddev;
    std::string pct_self;
};

struct pct_row
{
    std::string thread_id;
    std::string label;
    std::string count;
    std::string sum;
};

struct trip_row
{
    std::string thread_id;
    std::string depth;
    std::string label;
    std::string count;
};

// Split a string by tab.
static std::vector<std::string>
split_tabs(std::string const& line)
{
    std::vector<std::string> result;
    std::string              tok;
    for(char c : line)
    {
        if(c == '\t')
        {
            result.push_back(tok);
            tok.clear();
        }
        else
        {
            tok += c;
        }
    }
    result.push_back(tok);
    return result;
}

// Parse all non-comment, non-empty lines from a TSV stream into full-column rows.
static std::vector<tsv_row>
parse_full_rows(std::string const& tsv)
{
    std::vector<tsv_row> rows;
    std::istringstream   ss(tsv);
    std::string          line;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        auto cols = split_tabs(line);
        if(cols.size() < 11) continue;
        tsv_row r;
        r.thread_id = cols[0];
        r.depth     = cols[1];
        r.label     = cols[2];
        r.count     = cols[3];
        r.sum       = cols[4];
        r.mean      = cols[5];
        r.min_val   = cols[6];
        r.max_val   = cols[7];
        r.var       = cols[8];
        r.stddev    = cols[9];
        r.pct_self  = cols[10];
        rows.push_back(r);
    }
    return rows;
}

static std::vector<pct_row>
parse_pct_rows(std::string const& tsv)
{
    std::vector<pct_row> rows;
    std::istringstream   ss(tsv);
    std::string          line;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        auto cols = split_tabs(line);
        if(cols.size() < 4) continue;
        pct_row r;
        r.thread_id = cols[0];
        r.label     = cols[1];
        r.count     = cols[2];
        r.sum       = cols[3];
        rows.push_back(r);
    }
    return rows;
}

static std::vector<trip_row>
parse_trip_rows(std::string const& tsv)
{
    std::vector<trip_row> rows;
    std::istringstream    ss(tsv);
    std::string           line;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        auto cols = split_tabs(line);
        if(cols.size() < 4) continue;
        trip_row r;
        r.thread_id = cols[0];
        r.depth     = cols[1];
        r.label     = cols[2];
        r.count     = cols[3];
        rows.push_back(r);
    }
    return rows;
}

// Count comment header lines (lines starting with '#').
static int
count_header_lines(std::string const& tsv)
{
    std::istringstream ss(tsv);
    std::string        line;
    int                n = 0;
    while(std::getline(ss, line))
        if(!line.empty() && line[0] == '#') ++n;
    return n;
}

// Check that a header line containing key exists.
static bool
has_header_key(std::string const& tsv, std::string const& key)
{
    std::istringstream ss(tsv);
    std::string        line;
    while(std::getline(ss, line))
        if(line[0] == '#' && line.find(key) != std::string::npos) return true;
    return false;
}

// ── Sample construction helpers ───────────────────────────────────────────────

// Build a timer_sample with a given stack (innermost-last).
// beg_ns and end_ns determine the wall-clock duration.
static timer_sample
make_sample(int64_t tid, uint64_t beg_ns, uint64_t end_ns,
            std::vector<std::string> const& stack_labels)
{
    timer_sample s;
    s.tid    = tid;
    s.beg_ns = beg_ns;
    s.end_ns = end_ns;
    for(auto const& lbl : stack_labels)
    {
        stack_frame f;
        f.name = lbl;
        s.stack.push_back(f);
    }
    return s;
}

// ── Test 1: Header lines well-formed (empty input) ────────────────────────────

TEST(native_report_writer_tsv, header_present_with_zero_samples)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);
    writer.flush();

    auto wall_str = wall_out.str();
    EXPECT_GE(count_header_lines(wall_str), 3)
        << "wall_clock.tsv must have at least 3 header comment lines even with zero "
           "samples";
    EXPECT_TRUE(has_header_key(wall_str, "metric:"))
        << "wall_clock.tsv header must contain '# metric:' line";
    EXPECT_TRUE(has_header_key(wall_str, "unit:"))
        << "wall_clock.tsv header must contain '# unit:' line";
    EXPECT_TRUE(has_header_key(wall_str, "columns:"))
        << "wall_clock.tsv header must contain '# columns:' line";
}

TEST(native_report_writer_tsv, header_columns_match_l4_spec_wall_clock)
{
    // L4 spec: 11 columns for wall_clock.tsv:
    //   thread_id  depth  label  count  sum  mean  min  max  var  stddev  pct_self
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);
    writer.flush();

    auto wall_str = wall_out.str();
    // The columns: header line must contain all 11 column names.
    EXPECT_TRUE(has_header_key(wall_str, "thread_id")) << "missing thread_id in columns";
    EXPECT_TRUE(has_header_key(wall_str, "depth")) << "missing depth in columns";
    EXPECT_TRUE(has_header_key(wall_str, "label")) << "missing label in columns";
    EXPECT_TRUE(has_header_key(wall_str, "count")) << "missing count in columns";
    EXPECT_TRUE(has_header_key(wall_str, "sum")) << "missing sum in columns";
    EXPECT_TRUE(has_header_key(wall_str, "mean")) << "missing mean in columns";
    EXPECT_TRUE(has_header_key(wall_str, "min")) << "missing min in columns";
    EXPECT_TRUE(has_header_key(wall_str, "max")) << "missing max in columns";
    EXPECT_TRUE(has_header_key(wall_str, "var")) << "missing var in columns";
    EXPECT_TRUE(has_header_key(wall_str, "stddev")) << "missing stddev in columns";
    EXPECT_TRUE(has_header_key(wall_str, "pct_self")) << "missing pct_self in columns";
}

TEST(native_report_writer_tsv, header_present_with_zero_samples_percent_file)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);
    writer.flush();

    auto pct_str = pct_out.str();
    EXPECT_GE(count_header_lines(pct_str), 3)
        << "percent.tsv must have at least 3 header comment lines";
    EXPECT_TRUE(has_header_key(pct_str, "flat_scope"))
        << "percent.tsv header must document flat_scope semantics";
    EXPECT_TRUE(has_header_key(pct_str, "thread_id"))
        << "percent.tsv header must list column names";
}

TEST(native_report_writer_tsv, zero_samples_produces_no_data_rows)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    EXPECT_TRUE(rows.empty())
        << "zero samples must produce zero data rows in wall_clock.tsv";
}

// ── Test 2: Single-record round-trip ─────────────────────────────────────────

TEST(native_report_writer_tsv, single_sample_produces_one_data_row)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    // 1 second exactly: beg=0, end=1e9 ns.
    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "main" }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_EQ(rows.size(), 1U)
        << "one timer_sample with one stack frame must produce exactly one data row";
    EXPECT_EQ(rows[0].count, "1") << "count must be 1 for a single sample";
}

TEST(native_report_writer_tsv, single_sample_sum_equals_duration_seconds)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "main" }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].sum, "1.000000")
        << "sum must be 1.000000 s for a 1-second sample (6 decimal places, L4 #4)";
}

TEST(native_report_writer_tsv, single_sample_mean_equals_sum)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "main" }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].mean, rows[0].sum) << "mean must equal sum when count=1";
}

TEST(native_report_writer_tsv, single_sample_min_equals_max_equals_sum)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "main" }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].min_val, rows[0].sum) << "min must equal sum when count=1";
    EXPECT_EQ(rows[0].max_val, rows[0].sum) << "max must equal sum when count=1";
}

// ── Test 6: VAR=0 when n=1 (Bessel convention, L4 #8) ────────────────────────

TEST(native_report_writer_tsv, single_sample_var_is_zero_not_nan)
{
    // L4 #8: VAR=0 when n<2. Must NOT be NaN or empty.
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "main" }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].var, "0.000000")
        << "var must be 0.000000 (not NaN) when count=1 (Bessel n<2 convention, L4 #8)";
    EXPECT_EQ(rows[0].stddev, "0.000000")
        << "stddev must be 0.000000 (not NaN) when count=1";
}

// ── Test: leaf pct_self = 100 ─────────────────────────────────────────────────

TEST(native_report_writer_tsv, single_sample_leaf_pct_self_is_100)
{
    // A single-frame sample: the sole frame is a leaf; pct_self must be 100.
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0,
                               { make_sample(0, 0, 1'000'000'000ULL, { "leaf_func" }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    double pct = std::stod(rows[0].pct_self);
    EXPECT_NEAR(pct, 100.0, 0.01)
        << "a leaf frame (no children) must have pct_self = 100.0";
}

// ── Test 3: % SELF reconstruction over parent-child fixture ───────────────────
//
// 3 timer_samples:
//   s1: stack [A]           — duration 1s
//   s2: stack [A, B]        — duration 1s
//   s3: stack [A, B, C]     — duration 1s
//
// After aggregation per (depth, label):
//   depth=0, A: count=3, sum=3s, children include B(sum=2s) → pct_self = (3-2)/3 * 100
//   = 33.3% depth=1, B: count=2, sum=2s, children include C(sum=1s) → pct_self = (2-1)/2
//   * 100 = 50.0% depth=2, C: count=1, sum=1s, leaf                       → pct_self =
//   100.0%

TEST(native_report_writer_tsv, pct_self_parent_node_excludes_child_time)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t            sec     = 1'000'000'000ULL;
    std::vector<timer_sample> samples = {
        make_sample(0, 0 * sec, 1 * sec, { "A" }),
        make_sample(0, 1 * sec, 2 * sec, { "A", "B" }),
        make_sample(0, 2 * sec, 3 * sec, { "A", "B", "C" }),
    };

    writer.write_timer_samples(0, samples);
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());

    // Find the row for A at depth 0.
    auto it_a = std::find_if(rows.begin(), rows.end(), [](tsv_row const& r) {
        return r.depth == "0" && r.label == "A";
    });
    ASSERT_NE(it_a, rows.end()) << "must have a row for (depth=0, label=A)";
    EXPECT_EQ(it_a->count, "3") << "A appears in all 3 samples";
    double pct_a = std::stod(it_a->pct_self);
    EXPECT_NEAR(pct_a, 33.33, 1.0)
        << "A's pct_self must be ~33% (1 sample has no children at depth 1)";
}

TEST(native_report_writer_tsv, pct_self_middle_node_correct)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t            sec     = 1'000'000'000ULL;
    std::vector<timer_sample> samples = {
        make_sample(0, 0 * sec, 1 * sec, { "A" }),
        make_sample(0, 1 * sec, 2 * sec, { "A", "B" }),
        make_sample(0, 2 * sec, 3 * sec, { "A", "B", "C" }),
    };

    writer.write_timer_samples(0, samples);
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());

    auto it_b = std::find_if(rows.begin(), rows.end(), [](tsv_row const& r) {
        return r.depth == "1" && r.label == "B";
    });
    ASSERT_NE(it_b, rows.end()) << "must have a row for (depth=1, label=B)";
    EXPECT_EQ(it_b->count, "2") << "B appears in samples 2 and 3";
    double pct_b = std::stod(it_b->pct_self);
    EXPECT_NEAR(pct_b, 50.0, 1.0)
        << "B's pct_self must be ~50% (1 of 2 samples has no child C)";
}

TEST(native_report_writer_tsv, pct_self_leaf_node_is_100)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t            sec     = 1'000'000'000ULL;
    std::vector<timer_sample> samples = {
        make_sample(0, 0 * sec, 1 * sec, { "A" }),
        make_sample(0, 1 * sec, 2 * sec, { "A", "B" }),
        make_sample(0, 2 * sec, 3 * sec, { "A", "B", "C" }),
    };

    writer.write_timer_samples(0, samples);
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());

    auto it_c = std::find_if(rows.begin(), rows.end(), [](tsv_row const& r) {
        return r.depth == "2" && r.label == "C";
    });
    ASSERT_NE(it_c, rows.end()) << "must have a row for (depth=2, label=C)";
    double pct_c = std::stod(it_c->pct_self);
    EXPECT_NEAR(pct_c, 100.0, 0.01) << "C is a leaf; pct_self must be 100.0";
}

// ── Test 4: flat_scope for sampling_percent ───────────────────────────────────
//
// 'fib' appears at depth 5 in one sample and depth 6 in another.
// percent.tsv must show a SINGLE row for 'fib' (dedup'd), depth always 0.

TEST(native_report_writer_tsv, flat_scope_dedups_label_across_depths)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t sec = 1'000'000'000ULL;
    // Both stacks end with 'fib' at different depths.
    std::vector<std::string> stack_depth5 = { "a", "b", "c", "d", "e", "fib" };
    std::vector<std::string> stack_depth6 = { "a", "b", "c", "d", "e", "f", "fib" };

    writer.write_timer_samples(0, {
                                      make_sample(0, 0 * sec, 1 * sec, stack_depth5),
                                      make_sample(0, 1 * sec, 2 * sec, stack_depth6),
                                  });
    writer.flush();

    auto pct_rows = parse_pct_rows(pct_out.str());

    // Exactly one row for 'fib' (dedup across depths).
    int fib_count = 0;
    for(auto const& r : pct_rows)
        if(r.label == "fib") ++fib_count;

    EXPECT_EQ(fib_count, 1)
        << "flat_scope: 'fib' must appear in exactly one percent.tsv row "
           "regardless of call-stack depth";
}

TEST(native_report_writer_tsv, flat_scope_depth_column_always_zero)
{
    // percent.tsv has no depth column (it's omitted, or always 0 per flat_scope spec).
    // The 4-column format is: thread_id  label  count  sum.
    // We verify there is NO depth=5 or depth=6 field anywhere in the data rows.
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t           sec          = 1'000'000'000ULL;
    std::vector<std::string> stack_depth5 = { "a", "b", "c", "d", "e", "fib" };

    writer.write_timer_samples(0, {
                                      make_sample(0, 0 * sec, 1 * sec, stack_depth5),
                                  });
    writer.flush();

    auto pct_rows = parse_pct_rows(pct_out.str());
    ASSERT_FALSE(pct_rows.empty());

    // The second column in the percent file is label, not depth.
    // Verify that the "depth" position (col[1]) contains a label, not an integer >= 5.
    for(auto const& r : pct_rows)
    {
        bool looks_like_depth_int =
            !r.label.empty() && std::all_of(r.label.begin(), r.label.end(), ::isdigit) &&
            std::stoi(r.label) >= 2;
        EXPECT_FALSE(looks_like_depth_int)
            << "percent.tsv must not have a depth column — "
               "second column must be label, not a depth integer";
    }
}

TEST(native_report_writer_tsv, flat_scope_count_sums_across_depths)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t           sec          = 1'000'000'000ULL;
    std::vector<std::string> stack_depth5 = { "a", "b", "c", "d", "e", "fib" };
    std::vector<std::string> stack_depth6 = { "a", "b", "c", "d", "e", "f", "fib" };

    writer.write_timer_samples(0, {
                                      make_sample(0, 0 * sec, 1 * sec, stack_depth5),
                                      make_sample(0, 1 * sec, 2 * sec, stack_depth6),
                                  });
    writer.flush();

    auto pct_rows = parse_pct_rows(pct_out.str());
    auto it       = std::find_if(pct_rows.begin(), pct_rows.end(),
                                 [](pct_row const& r) { return r.label == "fib"; });
    ASSERT_NE(it, pct_rows.end()) << "must have a row for 'fib'";
    EXPECT_EQ(it->count, "2")
        << "flat_scope count must be sum across both stack depths where 'fib' appeared";
}

// ── Test 5: Multi-thread fan-out ──────────────────────────────────────────────

TEST(native_report_writer_tsv, multi_thread_all_tids_in_one_file)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t sec = 1'000'000'000ULL;
    for(int64_t tid : { 1, 2, 3 })
    {
        writer.write_timer_samples(tid,
                                   {
                                       make_sample(tid, 0 * sec, 1 * sec, { "func" }),
                                   });
    }
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());

    bool has_tid1 = false, has_tid2 = false, has_tid3 = false;
    for(auto const& r : rows)
    {
        if(r.thread_id == "1") has_tid1 = true;
        if(r.thread_id == "2") has_tid2 = true;
        if(r.thread_id == "3") has_tid3 = true;
    }
    EXPECT_TRUE(has_tid1) << "wall_clock.tsv must contain rows for tid=1";
    EXPECT_TRUE(has_tid2) << "wall_clock.tsv must contain rows for tid=2";
    EXPECT_TRUE(has_tid3) << "wall_clock.tsv must contain rows for tid=3";
}

TEST(native_report_writer_tsv, multi_thread_rows_distinguished_by_thread_id)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t sec = 1'000'000'000ULL;
    // Same label on different threads — must be separate rows.
    writer.write_timer_samples(1, { make_sample(1, 0, 1 * sec, { "shared_func" }) });
    writer.write_timer_samples(2, { make_sample(2, 0, 2 * sec, { "shared_func" }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());

    int tid1_rows = 0, tid2_rows = 0;
    for(auto const& r : rows)
    {
        if(r.label == "shared_func" && r.thread_id == "1") ++tid1_rows;
        if(r.label == "shared_func" && r.thread_id == "2") ++tid2_rows;
    }
    EXPECT_EQ(tid1_rows, 1) << "must have 1 row for tid=1, shared_func";
    EXPECT_EQ(tid2_rows, 1) << "must have 1 row for tid=2, shared_func";
}

// ── Test 7: Untruncated labels ────────────────────────────────────────────────

TEST(native_report_writer_tsv, label_is_not_truncated)
{
    // L4 #2: emit untruncated labels. No max_width.
    const std::string long_label =
        "tim::sampling::sampler<tim::lightweight_tuple<rocprofsys::component::"
        "backtrace_timestamp, rocprofsys::component::backtrace, "
        "rocprofsys::component::backtrace_metrics, rocprofsys::component::callchain>, 0>";

    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0,
                               { make_sample(0, 0, 1'000'000'000ULL, { long_label }) });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].label, long_label)
        << "label must be written untruncated (no '...' ellipsis, no max_width)";
    EXPECT_EQ(rows[0].label.find("..."), std::string::npos)
        << "label must contain no '...' truncation marker";
}

// ── Test 8: Numeric format precision ─────────────────────────────────────────

TEST(native_report_writer_tsv, numeric_precision_six_decimal_places)
{
    // L4 #4: 6 decimal places for wall/cpu columns.
    // 1'266'444 ns = 0.001266444 s → serialized as "0.001266" (truncated at 6dp).
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0, {
                                      make_sample(0, 0, 1'266'444ULL, { "precise_func" }),
                                  });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    // 0.001266444 s rounded to 6dp = 0.001266.
    EXPECT_EQ(rows[0].sum, "0.001266")
        << "wall_clock sum must use exactly 6 decimal places (L4 #4)";
}

// ── Test 10: Tab-only delimiter (no padding) ──────────────────────────────────

TEST(native_report_writer_tsv, delimiter_is_single_tab_no_padding)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "fn" }) });
    writer.flush();

    // Find first data row (non-comment).
    std::istringstream ss(wall_out.str());
    std::string        line;
    while(std::getline(ss, line))
        if(!line.empty() && line[0] != '#') break;

    ASSERT_FALSE(line.empty()) << "must have at least one data row";

    // No leading space.
    EXPECT_NE(line[0], ' ')
        << "data rows must not have leading spaces (tab-only delimiter, L4 #3)";

    // Columns are tab-separated, not space-padded.
    // A line with space-padding would contain multiple consecutive spaces.
    bool has_double_space = (line.find("  ") != std::string::npos);
    EXPECT_FALSE(has_double_space)
        << "data rows must not contain double spaces (fixed-width padding not permitted)";

    // Verify tab IS the delimiter.
    EXPECT_NE(line.find('\t'), std::string::npos)
        << "data rows must use tab as the column delimiter";
}

// ── trip_count.tsv: 4 columns (thread_id  depth  label  count) ───────────────

TEST(native_report_writer_tsv, trip_count_file_produced_with_correct_columns)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "main" }) });
    writer.flush();

    auto trip_str = trip_out.str();
    EXPECT_TRUE(has_header_key(trip_str, "thread_id"))
        << "trip_count.tsv missing thread_id";
    EXPECT_TRUE(has_header_key(trip_str, "depth")) << "trip_count.tsv missing depth";
    EXPECT_TRUE(has_header_key(trip_str, "label")) << "trip_count.tsv missing label";
    EXPECT_TRUE(has_header_key(trip_str, "count")) << "trip_count.tsv missing count";

    auto trip_rows = parse_trip_rows(trip_str);
    ASSERT_FALSE(trip_rows.empty())
        << "trip_count.tsv must have a data row for the single sample";
    EXPECT_EQ(trip_rows[0].count, "1") << "trip_count for a single sample must be 1";
    EXPECT_EQ(trip_rows[0].depth, "0")
        << "single-frame sample must be at depth 0 in trip_count";
}

// ── Multi-sample aggregation: count accumulates ───────────────────────────────

TEST(native_report_writer_tsv, two_samples_same_label_count_is_two)
{
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    const uint64_t sec = 1'000'000'000ULL;
    writer.write_timer_samples(0, {
                                      make_sample(0, 0 * sec, 1 * sec, { "func" }),
                                      make_sample(0, 1 * sec, 2 * sec, { "func" }),
                                  });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_EQ(rows.size(), 1U) << "two samples with the same label at the same depth "
                                  "must produce one aggregated row";
    EXPECT_EQ(rows[0].count, "2") << "count must be 2 for two samples of the same frame";
    EXPECT_EQ(rows[0].sum, "2.000000")
        << "sum must be 2.000000 s for two 1-second samples";
}

TEST(native_report_writer_tsv, two_samples_variance_computed_correctly)
{
    // Two samples: 1s and 3s.
    // Bessel-corrected: var = (sum_sq - sum²/n) / (n-1) = (10 - 16/2) / 1 = 2.0
    // stddev = sqrt(2.0) ≈ 1.414214
    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    writer.write_timer_samples(
        0, {
               make_sample(0, 0, 1'000'000'000ULL, { "func" }),
               make_sample(0, 1'000'000'000ULL, 4'000'000'000ULL, { "func" }),
           });
    writer.flush();

    auto rows = parse_full_rows(wall_out.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].var, "2.000000")
        << "Bessel-corrected variance for [1s,3s] must be exactly 2.000000";
    EXPECT_EQ(rows[0].stddev, "1.414214")
        << "stddev for [1s,3s] must be exactly 1.414214 (sqrt(2), 6dp)";
}

// ── C-17 regression: flush() must not crash when streams are partially/not set ─

TEST(native_report_writer_tsv, flush_with_no_streams_is_noop)
{
    // Default-constructed: no streams set. Must not crash or dereference null.
    native_report_writer writer;
    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "f" }) });
    EXPECT_NO_FATAL_FAILURE(writer.flush());
}

TEST(native_report_writer_tsv, flush_with_only_wall_stream_is_noop)
{
    // set_streams with one non-null pointer — flush must be a no-op (guard requires all
    // 4).
    native_report_writer writer;
    std::ostringstream   wall_out;
    writer.set_streams(&wall_out, nullptr, nullptr, nullptr);
    writer.write_timer_samples(0, { make_sample(0, 0, 1'000'000'000ULL, { "f" }) });
    EXPECT_NO_FATAL_FAILURE(writer.flush());
    // Data rows must NOT be written when the guard fires.
    auto data_rows = parse_full_rows(wall_out.str());
    EXPECT_TRUE(data_rows.empty())
        << "flush must emit nothing when any stream pointer is null";
}
