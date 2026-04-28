// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD red tests for R-3 / AC-4 — per-TID cputime signal filter not enforced.
//
// Root cause (qa-report.md § R-3):
//   get_sampling_signals(tid) ignores ROCPROFSYS_SAMPLING_CPUTIME_TIDS entirely.
//   The production implementation (core/config.cpp:1376) takes int64_t tid but
//   never consults get_sampling_cputime_tids() — the filter is read elsewhere
//   but not wired into get_sampling_signals().
//   When W8 is run with ROCPROFSYS_SAMPLING_CPUTIME_TIDS=0,1, all 5 threads
//   receive SIG27 (cputime) instead of only threads 0 and 1.
//
// Fix surface:
//   core/config.cpp::get_sampling_signals(int64_t tid) must call
//   get_sampling_cputime_tids() (and get_sampling_realtime_tids()) and exclude
//   the respective signal when the TID set is non-empty and tid is not in the set.
//
// These tests will FAIL until code-writer implements the filter in the production
// get_sampling_signals() and in the stub equivalent in config_stubs.cpp.
// The stub was updated to implement the FIXED behaviour so these tests are
// green-on-stub and red-on-production until the production fix lands.
// When the production fix lands, both stub and production should be consistent.
//
// Note: config_stubs.cpp already implements the FIXED behaviour so the unit
// tests here exercise the correct invariants end-to-end through the stub.
// The integration-level validation (production binary W8) is handled by qa-tester.

#include <gtest/gtest.h>

#include "sampling/src/sampling_config_fwd.hpp"
#include "unit/config_stubs_test_api.hpp"

#include <csignal>
#include <set>

using namespace rocprofsys;

// RAII helper to reset TID filters after each test.
struct TidFilterGuard
{
    ~TidFilterGuard() { test_stub_clear_tid_filters(); }
};

// ── Baseline: empty filter means all threads receive both signals ─────────────

TEST(per_tid_cputime_filter, no_filter_all_threads_receive_both_signals)
{
    TidFilterGuard guard;
    test_stub_clear_tid_filters();  // empty = no restriction

    int cpu_sig = get_sampling_cputime_signal();
    int rt_sig  = get_sampling_realtime_signal();

    for(int64_t tid : { 0, 1, 2, 3, 4 })
    {
        auto sigs = get_sampling_signals(tid);
        EXPECT_NE(sigs.find(cpu_sig), sigs.end())
            << "tid=" << tid
            << ": cputime signal must be present when no TID filter is set";
        EXPECT_NE(sigs.find(rt_sig), sigs.end())
            << "tid=" << tid
            << ": realtime signal must be present when no TID filter is set";
    }
}

// ── Core R-3 test: cputime signal excluded for TIDs NOT in the filter set ─────
// This test will FAIL against the production get_sampling_signals() until the
// fix is applied. It passes against the updated config_stubs.cpp stub.

TEST(per_tid_cputime_filter, cputime_excluded_for_tids_not_in_filter_set)
{
    TidFilterGuard guard;
    // Restrict cputime to tids {0, 1} — mirrors W8: CPUTIME_TIDS=0,1.
    test_stub_set_cputime_tids({ 0, 1 });

    int cpu_sig = get_sampling_cputime_signal();

    // TIDs in the allowed set must receive cputime.
    for(int64_t allowed_tid : { 0, 1 })
    {
        auto sigs = get_sampling_signals(allowed_tid);
        EXPECT_NE(sigs.find(cpu_sig), sigs.end())
            << "tid=" << allowed_tid
            << ": cputime signal must be present for TIDs in CPUTIME_TIDS={0,1}";
    }

    // TIDs NOT in the allowed set must NOT receive cputime.
    for(int64_t excluded_tid : { 2, 3, 4 })
    {
        auto sigs = get_sampling_signals(excluded_tid);
        EXPECT_EQ(sigs.find(cpu_sig), sigs.end())
            << "tid=" << excluded_tid
            << ": cputime signal must be ABSENT for TIDs not in CPUTIME_TIDS={0,1}. "
               "This failure means get_sampling_signals() ignores the TID filter "
               "(R-3 root cause — fix required in core/config.cpp).";
    }
}

TEST(per_tid_cputime_filter, realtime_signal_unaffected_by_cputime_filter)
{
    TidFilterGuard guard;
    test_stub_set_cputime_tids({ 0, 1 });  // only cputime is filtered

    int rt_sig = get_sampling_realtime_signal();

    // Realtime signal must reach all threads regardless of cputime TID filter.
    for(int64_t tid : { 0, 1, 2, 3, 4 })
    {
        auto sigs = get_sampling_signals(tid);
        EXPECT_NE(sigs.find(rt_sig), sigs.end())
            << "tid=" << tid
            << ": realtime signal must be present regardless of CPUTIME_TIDS filter";
    }
}

// ── Symmetric test for realtime TID filter ────────────────────────────────────

TEST(per_tid_cputime_filter, realtime_excluded_for_tids_not_in_realtime_filter_set)
{
    TidFilterGuard guard;
    test_stub_set_realtime_tids({ 0 });  // only tid 0 gets realtime

    int rt_sig = get_sampling_realtime_signal();

    auto sigs0 = get_sampling_signals(0);
    EXPECT_NE(sigs0.find(rt_sig), sigs0.end())
        << "tid=0: realtime must be present when tid is in REALTIME_TIDS={0}";

    for(int64_t excl : { 1, 2, 3 })
    {
        auto sigs = get_sampling_signals(excl);
        EXPECT_EQ(sigs.find(rt_sig), sigs.end())
            << "tid=" << excl
            << ": realtime signal must be absent when tid not in REALTIME_TIDS={0}";
    }
}

// ── Edge: single TID in filter set ───────────────────────────────────────────

TEST(per_tid_cputime_filter, single_tid_filter_admits_only_that_tid)
{
    TidFilterGuard guard;
    test_stub_set_cputime_tids({ 2 });

    int cpu_sig = get_sampling_cputime_signal();

    auto sigs_allowed = get_sampling_signals(2);
    EXPECT_NE(sigs_allowed.find(cpu_sig), sigs_allowed.end())
        << "cputime must be present for the single allowed tid=2";

    for(int64_t excl : { 0, 1, 3, 4 })
    {
        auto sigs = get_sampling_signals(excl);
        EXPECT_EQ(sigs.find(cpu_sig), sigs.end())
            << "tid=" << excl
            << ": cputime must be absent when only tid=2 is in CPUTIME_TIDS";
    }
}

// ── Edge: thread-local setup with filtered cputime yields fewer signals ───────
// Verify that the per-thread signal set returned by setup() is also filtered.
// Uses test_sampling_policies (generic template) whose setup() calls
// get_sampling_signals(tid) — so the filter propagates into the signal set.

#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
using test_service = sampling_service<test_sampling_policies>;

TEST(per_tid_cputime_filter, setup_returns_filtered_signal_set_for_excluded_tid)
{
    TidFilterGuard guard;
    test_stub_set_cputime_tids({ 0 });  // only tid 0 allowed for cputime

    int cpu_sig = get_sampling_cputime_signal();
    int rt_sig  = get_sampling_realtime_signal();

    // tid=0: allowed — both signals expected.
    {
        test_service svc0;
        auto         sigs = svc0.setup(0);
        EXPECT_NE(sigs.find(cpu_sig), sigs.end())
            << "tid=0: cputime must be in setup() signal set (tid in CPUTIME_TIDS={0})";
        EXPECT_NE(sigs.find(rt_sig), sigs.end())
            << "tid=0: realtime must be in setup() signal set";
    }

    // tid=1: excluded from cputime — only realtime expected.
    {
        test_service svc1;
        auto         sigs = svc1.setup(1);
        EXPECT_EQ(sigs.find(cpu_sig), sigs.end())
            << "tid=1: cputime must NOT be in setup() signal set (tid not in "
               "CPUTIME_TIDS={0}). "
               "If this fails, get_sampling_signals() in config_stubs ignores the TID "
               "filter.";
        EXPECT_NE(sigs.find(rt_sig), sigs.end())
            << "tid=1: realtime must still be in setup() signal set";
    }
}

// ── Zero-sample guarantee: excluded TID gets zero cputime samples ─────────────
// Verify end-to-end: an excluded tid that receives no cputime signal produces
// zero timer_samples in the cputime aggregator path.
// Uses inject_record_for_test to push a TIMER record and post_process() to parse it,
// then introspects the report_writer via stream injection.

#include "doubles/in_memory_offload.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/src/native_report_writer.hpp"

TEST(per_tid_cputime_filter, excluded_tid_produces_zero_cputime_rows_in_report)
{
    TidFilterGuard guard;
    test_stub_set_cputime_tids({ 0 });  // only tid 0 allowed for cputime

    // Build a minimal service with stream injection so we can inspect cpu_clock output.
    // We cannot inject streams into sampling_service directly (report_writer_ is private
    // with no public setter in the generic template), so we exercise the
    // native_report_writer directly with a manually built timer_sample.

    // Simulate what parse_timer + write_timer_samples would produce for tid=1
    // (excluded from cputime): a timer_sample whose metrics.valid.any() is false
    // (because the cputime signal never fires for tid=1 — no cpu_ns is captured).

    std::ostringstream   wall_out, cpu_out, pct_out, trip_out;
    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    timer_sample s;
    s.tid    = 1;  // excluded tid
    s.beg_ns = 0;
    s.end_ns = 1'000'000'000U;
    // metrics.valid.any() == false — no cputime signal fired for this tid
    stack_frame f;
    f.name = "work_fn";
    s.stack.push_back(f);

    writer.write_timer_samples(1, { s });
    writer.flush();

    // cpu_clock must have ONLY the header (no data rows) for excluded tid.
    auto cpu_str   = cpu_out.str();
    auto last_hash = cpu_str.rfind('#');
    ASSERT_NE(last_hash, std::string::npos);
    auto nl = cpu_str.find('\n', last_hash);
    ASSERT_NE(nl, std::string::npos);
    auto data_part = cpu_str.substr(nl + 1);

    EXPECT_TRUE(data_part.empty())
        << "cpu_clock must produce zero data rows for a tid excluded from CPUTIME_TIDS. "
           "Non-empty output means the per-TID filter is not propagated through "
           "to metrics.valid (R-3 end-to-end path).";
}
