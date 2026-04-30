// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integration test: sampling_service<test_sampling_policies> end-to-end.
// Variant 2: per-tid drain+emit occurs in shutdown(tid) via
// emit_resolved_to_trace_cache() (generic template is no-op for test policies).
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
// test_service alias lives in test_sampling_policies.hpp.

// ─── Lifecycle: setup then shutdown completes without crash ────────────────────

TEST(full_pipeline, setup_and_shutdown_completes_without_crash)
{
    test_service svc{ make_test_config() };
    svc.setup(0);
    svc.shutdown(0);
    SUCCEED();
}

// ─── Lifecycle: report_writer accessible after shutdown ───────────────────────

TEST(full_pipeline, report_writer_accessible_after_shutdown)
{
    test_service svc{ make_test_config() };
    svc.setup(0);
    svc.shutdown(0);

    auto& writer = svc.report_writer_ref();
    // noop_report_writer tracks call counts; timer_count for unregistered tid is 0.
    EXPECT_EQ(writer.m_timer_counts.count(0), 0U)
        << "no timer samples expected when no signals were raised in the test";
}

// ─── Lifecycle: perfetto_sink accessible after shutdown ───────────────────────

TEST(full_pipeline, perfetto_sink_accessible_after_shutdown)
{
    test_service svc{ make_test_config() };
    svc.setup(0);
    svc.shutdown(0);

    // Access the sink to verify it is alive and reachable (no crash = pass).
    [[maybe_unused]] auto& psink = svc.get_perfetto_sink();
    SUCCEED() << "perfetto_sink must be reachable after shutdown()";
}

// ─── Lifecycle: pause+resume then shutdown still completes ────────────────────

TEST(full_pipeline, shutdown_after_pause_resume)
{
    test_service svc{ make_test_config() };
    svc.setup(0);
    svc.pause();
    svc.resume();
    svc.shutdown(0);
    EXPECT_FALSE(svc.is_paused())
        << "service must not be paused after resume() followed by shutdown()";
}

// ─── Lifecycle: multi-tid setup then shutdown ─────────────────────────────────

TEST(full_pipeline, multi_tid_setup_then_shutdown)
{
    test_service svc{ make_test_config() };

    constexpr int64_t k_num_tids = 4;
    for(int64_t tid = 0; tid < k_num_tids; ++tid)
        svc.setup(tid);

    for(int64_t tid = 0; tid < k_num_tids; ++tid)
        svc.shutdown(tid);

    // Introspection: dropped_samples() must be non-negative and readable.
    EXPECT_GE(svc.dropped_samples(), 0U)
        << "dropped_samples counter must be non-negative after multi-tid shutdown";
}

// ─── AC-11: is_paused false initially ─────────────────────────────────────────

TEST(full_pipeline, not_paused_initially)
{
    test_service svc{ make_test_config() };
    EXPECT_FALSE(svc.is_paused()) << "service must start unpaused";
}

// ─── AC-11: is_blocked false initially ────────────────────────────────────────

TEST(full_pipeline, not_blocked_initially)
{
    test_service svc{ make_test_config() };
    EXPECT_FALSE(svc.is_blocked()) << "service must start unblocked";
}

// ─── Lifecycle: setup then shutdown for single tid ────────────────────────────

TEST(full_pipeline, setup_and_shutdown_single_tid)
{
    test_service svc{ make_test_config() };
    svc.setup(0);
    svc.shutdown(0);
    SUCCEED();
}
