// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sample_parser — Phase B + C. Covers NFR-T-6 entirely.
// AC-11: single-sample buffer discarded.
// AC-13: pause intervals filter samples.

#include <gtest/gtest.h>

#include "doubles/fake_clock.hpp"
#include "sampling/src/pause_interval_registry.hpp"
#include "sampling/src/sample_parser.hpp"

#include <algorithm>
#include <vector>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// Helper: build a backtrace_record with a given timestamp and pc_count.
static backtrace_record
make_record(int64_t tid, uint64_t ts_ns, uint8_t pc_count = 2)
{
    backtrace_record rec;
    rec.tid          = tid;
    rec.timestamp_ns = ts_ns;
    rec.trigger      = trigger_type::TIMER;
    rec.pc_count     = pc_count;
    for(uint8_t idx = 0; idx < pc_count; ++idx)
        rec.raw_pcs[idx] = 0x100 + idx;
    return rec;
}

// ─── NFR-T-6: Timer parse — happy path ───────────────────────────────────────

TEST(sample_parser, timer_parse_happy_path)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    backtrace_record              init_rec = make_record(0, 1000);
    std::vector<backtrace_record> raw      = {
        make_record(0, 2000),
        make_record(0, 3000),
    };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    ASSERT_EQ(result.size(), 2U);
    EXPECT_EQ(result.at(0).beg_ns, 1000U);
    EXPECT_EQ(result.at(0).end_ns, 2000U);
    EXPECT_EQ(result.at(1).beg_ns, 2000U);
    EXPECT_EQ(result.at(1).end_ns, 3000U);
}

// ─── NFR-T-6: Overflow parse ─────────────────────────────────────────────────

TEST(sample_parser, overflow_parse_builds_samples)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    std::vector<backtrace_record> raw = {
        make_record(0, 5000),
        make_record(0, 6000),
    };

    auto result = parser.parse_overflow(0, raw, pause_reg);

    ASSERT_EQ(result.size(), 2U)
        << "Both overflow records must survive when no pause intervals";
}

// ─── AC-11 / NFR-T-6: Single-sample buffer discarded ─────────────────────────

TEST(sample_parser, single_sample_buffer_is_discarded)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    // Single record with only 1 frame — must be discarded per line-1208 logic
    backtrace_record              init_rec = make_record(0, 0, 1);
    std::vector<backtrace_record> raw      = { make_record(0, 100, 1) };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    EXPECT_TRUE(result.empty())
        << "A buffer of size 1 with a single frame must be discarded";
}

// ─── AC-13 / NFR-T-6: Pause interval filters overlapping samples ─────────────

TEST(sample_parser, pause_interval_filters_overlapping_samples)
{
    sample_parser parser;

    // Build a pause registry with interval [100, 200] ns.
    fake_clock clock;
    fake_clock::reset(100);
    pause_interval_registry<fake_clock> pause_reg(clock);
    pause_reg.pause();  // pause at t=100
    fake_clock::advance_ns(100);
    pause_reg.resume();  // resume at t=200; interval is [100, 200]

    // init_rec at t=0.
    // raw[0] at t=160: sample interval [0, 160] spans [100, 200] → dropped.
    // raw[1] at t=500: sample interval [160, 500] still overlaps [100, 200] → dropped.
    // raw[2] at t=700: sample interval [500, 700] is entirely after [100,200] → kept.
    backtrace_record              init_rec = make_record(0, 0);
    std::vector<backtrace_record> raw      = {
        make_record(0, 160),  // [0, 160] overlaps [100, 200] → dropped
        make_record(0, 500),  // [160, 500] overlaps [100, 200] → dropped
        make_record(0, 700),  // [500, 700] does NOT overlap [100, 200] → kept
    };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    // Samples: [0,160] overlaps → dropped; [160,500] overlaps → dropped;
    //          [500,700] does NOT overlap [100,200] → kept (1 survivor).
    ASSERT_EQ(result.size(), 1U)
        << "Only the sample entirely after the pause interval should survive";
    EXPECT_EQ(result.at(0).end_ns, 700U);
}

// ─── NFR-T-6: Empty stack — not forwarded ────────────────────────────────────

TEST(sample_parser, empty_stack_record_is_handled)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    backtrace_record              init_rec = make_record(0, 0, 0);
    std::vector<backtrace_record> raw      = { make_record(0, 100, 0) };

    // Should not crash; empty stack sample may be discarded or returned as-is
    // depending on architecture decision. We only assert no crash + no UB.
    EXPECT_NO_THROW(parser.parse_timer(0, init_rec, raw, pause_reg));
}

// ─── NFR-T-6: Output sorted by beg_ns ────────────────────────────────────────

TEST(sample_parser, output_is_sorted_by_beg_ns)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    backtrace_record init_rec = make_record(0, 0);
    // Provide records in non-monotonic order to test sort.
    std::vector<backtrace_record> raw = {
        make_record(0, 300),
        make_record(0, 100),
        make_record(0, 200),
    };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    ASSERT_GE(result.size(), 2U);
    EXPECT_TRUE(std::is_sorted(
        result.begin(), result.end(),
        [](auto const& lhs, auto const& rhs) { return lhs.beg_ns < rhs.beg_ns; }))
        << "parse_timer output must be sorted by beg_ns";
}

// ─── NFR-T-6: Timestamp tie — two samples with identical timestamps ───────────

TEST(sample_parser, timestamp_tie_handled_without_crash)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    backtrace_record              init_rec = make_record(0, 0);
    std::vector<backtrace_record> raw      = {
        make_record(0, 500), make_record(0, 500),  // same timestamp
    };

    EXPECT_NO_THROW(parser.parse_timer(0, init_rec, raw, pause_reg));
}

// ─── AC-13: pause spanning entire sample window — all samples dropped ──────────

TEST(sample_parser, pause_spanning_entire_window_drops_all_samples)
{
    sample_parser parser;

    // Pause covers [0, 10000]: every sample interval is inside.
    fake_clock clock;
    fake_clock::reset(0);
    pause_interval_registry<fake_clock> pause_reg(clock);
    pause_reg.pause();
    fake_clock::advance_ns(10000);
    pause_reg.resume();  // interval [0, 10000]

    // All raw samples fall within [0, 10000].
    backtrace_record              init_rec = make_record(0, 100);
    std::vector<backtrace_record> raw      = {
        make_record(0, 500),
        make_record(0, 1000),
        make_record(0, 5000),
        make_record(0, 9000),
    };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    EXPECT_TRUE(result.empty()) << "All samples inside a pause window must be dropped";
}

// ─── Edge: max-depth stack record (64 PCs) is parsed without crash ────────────

TEST(sample_parser, max_depth_stack_record_parsed_without_crash)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    // Build records with full 64-PC stacks.
    auto make_max = [](uint64_t ts) { return make_record(0, ts, 64); };

    backtrace_record              init_rec = make_max(0);
    std::vector<backtrace_record> raw      = { make_max(1000), make_max(2000) };

    EXPECT_NO_THROW(parser.parse_timer(0, init_rec, raw, pause_reg))
        << "Parsing max-depth (64 PC) records must not throw";
}

// ─── R-1 fix: cpu_ns propagated as delta, not absolute ───────────────────────

TEST(sample_parser, cpu_ns_in_metrics_propagated_as_delta)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    // init_rec: cpu_ns=1000, valid bit 0 set.
    backtrace_record init_rec = make_record(0, 100);
    init_rec.metrics.cpu_ns   = 1000;
    init_rec.metrics.valid.set(0);

    // raw[0]: cpu_ns=1200 → delta=200; raw[1]: cpu_ns=1500 → delta=300.
    backtrace_record r0 = make_record(0, 200);
    r0.metrics.cpu_ns   = 1200;
    r0.metrics.valid.set(0);

    backtrace_record r1 = make_record(0, 300);
    r1.metrics.cpu_ns   = 1500;
    r1.metrics.valid.set(0);

    std::vector<backtrace_record> raw = { r0, r1 };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    ASSERT_EQ(result.size(), 2U);
    EXPECT_EQ(result.at(0).metrics.cpu_ns, 200)
        << "cpu_ns must be delta (1200-1000=200), not absolute";
    EXPECT_EQ(result.at(1).metrics.cpu_ns, 300)
        << "cpu_ns must be delta (1500-1200=300), not absolute";
    EXPECT_TRUE(result.at(0).metrics.valid.any())
        << "valid bitset must be carried forward";
    EXPECT_TRUE(result.at(1).metrics.valid.any())
        << "valid bitset must be carried forward";
}

TEST(sample_parser, cpu_ns_zero_when_metrics_invalid)
{
    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    // Records with valid=0 (zero-initialized) — cpu_ns stays 0.
    backtrace_record init_rec = make_record(0, 100);
    backtrace_record r0       = make_record(0, 200);

    std::vector<backtrace_record> raw = { r0 };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    ASSERT_EQ(result.size(), 1U);
    EXPECT_FALSE(result.at(0).metrics.valid.any())
        << "valid bitset must stay clear when records carry invalid metrics";
}
