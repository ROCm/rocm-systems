// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD red tests for R-1 (cpu_clock always empty).
//
// Root cause (qa-report.md § R-1):
//   Part A — signal handler never calls clock_gettime(CLOCK_THREAD_CPUTIME_ID,...);
//             rec.metrics stays zero-initialized; metrics.valid.any() is always false.
//   Part B — parse_timer() copies absolute cpu_ns from each record, not the
//             per-interval delta (analogous to beg_ns/end_ns computation).
//   Part C — native_report_writer gates cpu_clock accumulation on metrics.valid.any();
//             with Parts A+B unfix'd this gate is never true.
//
// These tests verify the FIXED behaviour; they will FAIL until code-writer
// implements Parts A and B in services_accessor.cpp and sample_parser_impl.hpp.
//
// Tests are independent of the production signal handler — they exercise
// the data structures and parser directly so no POSIX timer is needed.

#include <gtest/gtest.h>

#include "doubles/fake_clock.hpp"
#include "sampling/data/backtrace_metrics_data.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/src/pause_interval_registry.hpp"
#include "sampling/src/sample_parser.hpp"

#include <cstdint>
#include <sstream>
#include <vector>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// ── Part A fixtures ───────────────────────────────────────────────────────────
// These tests verify backtrace_metrics_data semantics — that a record whose
// cpu_ns is set and valid bit 0 is set satisfies valid.any().
// The signal handler MUST populate these fields; if it doesn't, valid.any() is
// always false and the cpu_clock file is empty.

TEST(metrics_population, valid_bitset_is_false_when_zero_initialized)
{
    backtrace_metrics_data m{};
    EXPECT_FALSE(m.valid.any())
        << "zero-initialized metrics must not report valid (handler must populate)";
}

TEST(metrics_population, valid_bitset_is_true_after_setting_bit0)
{
    backtrace_metrics_data m{};
    m.cpu_ns = 500'000'000LL;
    m.valid.set(0);
    EXPECT_TRUE(m.valid.any())
        << "metrics.valid.any() must be true after setting bit 0 (cpu_ns valid)";
    EXPECT_EQ(m.cpu_ns, 500'000'000LL)
        << "cpu_ns must retain the value set on the metrics struct";
}

TEST(metrics_population, cpu_ns_is_zero_in_uninitialised_record)
{
    backtrace_record rec{};
    EXPECT_EQ(rec.metrics.cpu_ns, 0)
        << "freshly constructed backtrace_record must have cpu_ns == 0 "
           "(signal handler must explicitly populate it via clock_gettime)";
    EXPECT_FALSE(rec.metrics.valid.any())
        << "freshly constructed backtrace_record must have valid.any() == false";
}

// ── Part B fixtures ───────────────────────────────────────────────────────────
// parse_timer() must compute per-interval cpu_ns deltas (analogous to
// beg_ns/end_ns). A series of records with monotonically increasing cpu_ns must
// produce timer_samples whose metrics.cpu_ns == (rec[i].cpu_ns - rec[i-1].cpu_ns)
// and metrics.valid must be propagated from the raw record.

static backtrace_record
make_cpu_record(uint64_t wall_ns, int64_t cpu_ns)
{
    backtrace_record r{};
    r.tid            = 0;
    r.timestamp_ns   = wall_ns;
    r.trigger        = trigger_type::TIMER;
    r.pc_count       = 2;
    r.raw_pcs[0]     = 0x1000;
    r.raw_pcs[1]     = 0x2000;
    r.metrics.cpu_ns = cpu_ns;
    r.metrics.valid.set(0);  // bit 0 = cpu_ns valid
    return r;
}

TEST(metrics_population, parse_timer_propagates_metrics_valid_to_timer_sample)
{
    // After Part A fix: records carry non-zero cpu_ns and valid bit 0 set.
    // After Part B fix: parse_timer() computes delta and sets metrics on each sample.
    // This test will FAIL until both parts are fixed.

    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    // Simulate: signal handler captured cpu_ns at each delivery.
    // init_rec at wall=0, cpu=0 (first delivery, baseline).
    // raw[0] at wall=1000, cpu=400   → interval cpu_ns = 400-0 = 400
    // raw[1] at wall=2000, cpu=900   → interval cpu_ns = 900-400 = 500
    backtrace_record              init_rec = make_cpu_record(0, 0);
    std::vector<backtrace_record> raw      = {
        make_cpu_record(1000, 400),
        make_cpu_record(2000, 900),
    };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);

    ASSERT_EQ(result.size(), 2U)
        << "parse_timer must return 2 samples for 2 raw records (init + 2 tail)";

    // Part B: deltas, not absolute values.
    EXPECT_TRUE(result[0].metrics.valid.any())
        << "timer_sample[0].metrics.valid must be true after Part A+B fix";
    EXPECT_EQ(result[0].metrics.cpu_ns, 400)
        << "timer_sample[0].metrics.cpu_ns must equal the delta (400-0=400)";

    EXPECT_TRUE(result[1].metrics.valid.any())
        << "timer_sample[1].metrics.valid must be true after Part A+B fix";
    EXPECT_EQ(result[1].metrics.cpu_ns, 500)
        << "timer_sample[1].metrics.cpu_ns must equal the delta (900-400=500)";
}

TEST(metrics_population, parse_timer_cpu_ns_absolute_copy_is_not_a_delta)
{
    // Regression guard: documents the CURRENT (broken) behaviour.
    // With the broken parse_timer() (s.metrics = rec.metrics; no delta),
    // the second sample would have cpu_ns == 900 (absolute), not 500 (delta).
    // After the fix, this test asserts the CORRECT behaviour: delta == 500.
    //
    // This test duplicates the assertion above to make the broken vs fixed
    // values explicit in the test output on failure.

    sample_parser                       parser;
    fake_clock                          clock;
    pause_interval_registry<fake_clock> pause_reg(clock);

    backtrace_record              init_rec = make_cpu_record(0, 0);
    std::vector<backtrace_record> raw      = {
        make_cpu_record(1000, 400),
        make_cpu_record(2000, 900),
    };

    auto result = parser.parse_timer(0, init_rec, raw, pause_reg);
    ASSERT_EQ(result.size(), 2U);

    // The fixed parser MUST return a delta, not the absolute cpu_ns.
    // If this fails with cpu_ns == 900 instead of 500, the fix is missing.
    EXPECT_NE(result[1].metrics.cpu_ns, 900)
        << "parse_timer must NOT copy absolute cpu_ns (900); it must compute delta "
           "(500). "
           "A value of 900 indicates the Part B fix is missing.";
    EXPECT_EQ(result[1].metrics.cpu_ns, 500)
        << "parse_timer must produce a delta cpu_ns of 500 (= 900 - 400)";
}

// ── Part C fixture ────────────────────────────────────────────────────────────
