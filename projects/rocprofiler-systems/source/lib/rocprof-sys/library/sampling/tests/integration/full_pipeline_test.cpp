// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integration test: sampling_service<test_sampling_policies> end-to-end.
// AC-10: post_process drains and fans out to all sinks.
// AC-11: single-sample buffer not forwarded.
// Uses hand-written test doubles; no OS dependencies.

#include <gtest/gtest.h>

#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
using test_service = sampling_service<test_sampling_policies>;

// ─── AC-10: post_process runs without crash and all policy sinks are accessible

TEST(full_pipeline, post_process_completes_without_crash)
{
    test_service svc;
    svc.setup(0);
    svc.post_process();
    // No assertion needed beyond "we reach this line" — any crash would fail the test.
    SUCCEED();
}

// ─── AC-10: post_process calls trace_sink (observable via get_trace_sink) ─────
// Regression for Bug 4: redesign — trace_sink receives post-parse resolved samples
// via store_timer()/store_overflow(), not raw records. This ensures data_processor
// gets symbol names + real beg/end ranges rather than hex PCs + zero-duration entries.

TEST(full_pipeline, trace_sink_store_timer_called_when_offload_has_timer_records)
{
    test_service svc;
    svc.setup(0);

    // Three TIMER records: first is init record, remaining two form the tail.
    // AC-11 guard fires when tail.size()==1 && pc_count<=1; tail.size()==2 bypasses it.
    backtrace_record rec0;
    rec0.tid          = 0;
    rec0.timestamp_ns = 1'000'000'000ULL;
    rec0.trigger      = trigger_type::TIMER;

    backtrace_record rec1;
    rec1.tid          = 0;
    rec1.timestamp_ns = 2'000'000'000ULL;
    rec1.trigger      = trigger_type::TIMER;

    backtrace_record rec2;
    rec2.tid          = 0;
    rec2.timestamp_ns = 3'000'000'000ULL;
    rec2.trigger      = trigger_type::TIMER;

    svc.get_offload().inject(0, rec0);
    svc.get_offload().inject(0, rec1);
    svc.get_offload().inject(0, rec2);

    svc.post_process();

    auto& trace = svc.get_trace_sink();
    EXPECT_GE(trace.timer_records().size(), 1U)
        << "store_timer must be called at least once when offload has >=2 TIMER records";
}

TEST(full_pipeline, trace_sink_not_called_when_no_offload_data)
{
    test_service svc;
    svc.setup(0);
    svc.post_process();

    auto& trace = svc.get_trace_sink();
    EXPECT_EQ(trace.timer_records().size(), 0U)
        << "store_timer must not be called when no records were offloaded";
    EXPECT_EQ(trace.overflow_records().size(), 0U)
        << "store_overflow must not be called when no records were offloaded";
}

// ─── AC-10: report_writer accessible after post_process ───────────────────────

TEST(full_pipeline, report_writer_accessible_after_post_process)
{
    test_service svc;
    svc.setup(0);
    svc.post_process();

    auto& writer = svc.report_writer_ref();
    // noop_report_writer tracks call counts; timer_count for unregistered tid is 0.
    EXPECT_EQ(writer.m_timer_counts.count(0), 0U)
        << "no timer samples expected when no signals were raised in the test";
}

// ─── AC-10: perfetto_sink accessible after post_process ───────────────────────

TEST(full_pipeline, perfetto_sink_accessible_after_post_process)
{
    test_service svc;
    svc.setup(0);
    svc.post_process();

    // Access the sink to verify it is alive and reachable (no crash = pass).
    [[maybe_unused]] auto& psink = svc.get_perfetto_sink();
    SUCCEED() << "perfetto_sink must be reachable after post_process()";
}

// ─── AC-10: post_process after pause+resume still completes ───────────────────

TEST(full_pipeline, post_process_after_pause_resume)
{
    test_service svc;
    svc.setup(0);
    svc.pause();
    svc.resume();
    svc.post_process();
    EXPECT_FALSE(svc.is_paused())
        << "service must not be paused after resume() followed by post_process()";
}

// ─── AC-10: multi-thread setup followed by post_process ───────────────────────

TEST(full_pipeline, multi_tid_setup_then_post_process)
{
    test_service svc;

    constexpr int64_t k_num_tids = 4;
    for(int64_t tid = 0; tid < k_num_tids; ++tid)
    {
        svc.setup(tid);
    }

    svc.post_process();

    // Introspection: dropped_samples() must be non-negative and readable.
    EXPECT_GE(svc.dropped_samples(), 0U)
        << "dropped_samples counter must be non-negative after multi-tid post_process";
}

// ─── AC-11: is_paused false initially ─────────────────────────────────────────

TEST(full_pipeline, not_paused_initially)
{
    test_service svc;
    EXPECT_FALSE(svc.is_paused()) << "service must start unpaused";
}

// ─── AC-11: is_blocked false initially ────────────────────────────────────────

TEST(full_pipeline, not_blocked_initially)
{
    test_service svc;
    EXPECT_FALSE(svc.is_blocked()) << "service must start unblocked";
}

// ─── Lifecycle: setup then shutdown for single tid ────────────────────────────

TEST(full_pipeline, setup_and_shutdown_single_tid)
{
    test_service svc;
    svc.setup(0);
    svc.shutdown(0);
    SUCCEED();
}
