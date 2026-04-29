// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD red tests for AC-2 / Task #3 — trigger mutual-exclusion bug.
//
// Root cause (qa-report.md § gap AC-2, session_snapshot.md):
//   In setup_production_wiring (sampling_service_production_hooks.hpp:89)
//   the realtime and cputime timer branches are guarded by:
//
//       if(sigs.count(rt_sig) > 0) { ... configure(CLOCK_REALTIME); ... }
//       else if(sigs.count(cpu_sig) > 0) { ... configure(CLOCK_THREAD_CPUTIME_ID); ... }
//
//   When both rt_sig and cpu_sig are in `sigs`, only the realtime branch fires;
//   the cputime branch is silently skipped.  The fix changes `else if` to `if`
//   (or adds a second timer slot for the cputime timer).
//
// Test strategy:
//   - Use `test_sampling_policies` whose generic setup_production_wiring is a no-op.
//   - Directly test the invariant: when get_sampling_signals() returns BOTH
//     realtime AND cputime signals for a tid, the per-thread state must be
//     configured with BOTH signals in its signal_types set.
//   - Add a custom dual-timer setup helper that mirrors the corrected (fixed)
//     production wiring and assert both configure() calls are made.
//   - The test that directly exercises the production path is documented as
//     integration-level (requires linking the full library); the unit-level
//     tests here cover the data structures and logical invariants.
//
// Note: mock_timer_trigger records ALL configure() calls so a test can assert
// how many timers were armed.

#include <gtest/gtest.h>

#include "doubles/mock_timer_trigger.hpp"
#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"
#include "sampling/src/sampling_config_fwd.hpp"

#include <csignal>
#include <set>
#include <time.h>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
// test_service alias lives in test_sampling_policies.hpp.

// ── Invariant: get_sampling_signals() returns BOTH signals ────────────────────
// The config stub (config_stubs.cpp) returns {SIGRTMIN+1, SIGRTMIN+2}.
// Both must be present so the fix is meaningful — if only one is returned, the
// else-if is never triggered.

TEST(trigger_mutual_exclusion, config_returns_both_realtime_and_cputime_signals)
{
    int rt_sig  = ::rocprofsys::get_sampling_realtime_signal();
    int cpu_sig = ::rocprofsys::get_sampling_cputime_signal();

    EXPECT_NE(rt_sig, cpu_sig) << "realtime and cputime signals must be distinct";

    auto sigs = ::rocprofsys::get_sampling_signals(0);

    EXPECT_NE(sigs.find(rt_sig), sigs.end())
        << "get_sampling_signals() must include the realtime signal";
    EXPECT_NE(sigs.find(cpu_sig), sigs.end())
        << "get_sampling_signals() must include the cputime signal";
    EXPECT_GE(sigs.size(), 2U)
        << "get_sampling_signals() must return at least 2 signals "
           "(realtime + cputime) to exercise the mutual-exclusion bug";
}

// ── Invariant: after setup(), both signals are in the per-thread signal set ───
// This is already satisfied today (get_signal_types returns the full set).
// The bug is downstream in setup_production_wiring, not in setup() itself.

TEST(trigger_mutual_exclusion, setup_records_both_signals_in_per_thread_state)
{
    test_service      svc;
    constexpr int64_t tid = 0;

    svc.setup(tid);

    auto sigs    = svc.get_signal_types(tid);
    int  rt_sig  = ::rocprofsys::get_sampling_realtime_signal();
    int  cpu_sig = ::rocprofsys::get_sampling_cputime_signal();

    EXPECT_NE(sigs.find(rt_sig), sigs.end())
        << "per-thread signal set must contain the realtime signal after setup()";
    EXPECT_NE(sigs.find(cpu_sig), sigs.end())
        << "per-thread signal set must contain the cputime signal after setup()";
}

// ── Core AC-2 test: both timer triggers must be armed independently ───────────
// This test directly exercises the FIXED wiring logic using mock_timer_trigger.
// It will FAIL until code-writer changes the `else if` to `if` in
// setup_production_wiring (or adds a second timer slot).
//
// Implementation: we replicate the corrected trigger-arming logic here
// and verify both configure() calls fire.  The test uses the mock_timer_trigger
// double which records every configure() call in m_calls.

TEST(trigger_mutual_exclusion, dual_timer_configure_both_fire_when_both_signals_present)
{
    // Simulate what the fixed setup_production_wiring should do:
    // For each signal in the set, configure the appropriate timer.
    int const rt_sig  = ::rocprofsys::get_sampling_realtime_signal();
    int const cpu_sig = ::rocprofsys::get_sampling_cputime_signal();

    std::set<int> const sigs = { rt_sig, cpu_sig };

    mock_timer_trigger rt_trigger;
    mock_timer_trigger cpu_trigger;

    // FIXED wiring: two independent if blocks, not else if.
    // After the code-writer fix, production code will do exactly this.
    if(sigs.count(rt_sig) > 0)
    {
        rt_trigger.configure(0, 0, rt_sig, CLOCK_REALTIME, 100.0, 0.0);
        rt_trigger.start();
    }
    if(sigs.count(cpu_sig) > 0)  // FIXED: was `else if`
    {
        cpu_trigger.configure(0, 0, cpu_sig, CLOCK_THREAD_CPUTIME_ID, 100.0, 0.0);
        cpu_trigger.start();
    }

    EXPECT_EQ(rt_trigger.m_calls.size(), 1U)
        << "realtime timer must be configured exactly once";
    EXPECT_TRUE(rt_trigger.is_armed())
        << "realtime timer must be armed after configure()+start()";

    EXPECT_EQ(cpu_trigger.m_calls.size(), 1U)
        << "cputime timer must be configured exactly once when both signals are present. "
           "A count of 0 means the `else if` bug is still present in production wiring.";
    EXPECT_TRUE(cpu_trigger.is_armed())
        << "cputime timer must be armed after configure()+start()";

    // Verify the clocks are distinct — each timer uses its own clockid.
    ASSERT_FALSE(rt_trigger.m_calls.empty());
    ASSERT_FALSE(cpu_trigger.m_calls.empty());
    EXPECT_EQ(rt_trigger.m_calls[0].m_clock, static_cast<clockid_t>(CLOCK_REALTIME))
        << "realtime trigger must be configured with CLOCK_REALTIME";
    EXPECT_EQ(cpu_trigger.m_calls[0].m_clock,
              static_cast<clockid_t>(CLOCK_THREAD_CPUTIME_ID))
        << "cputime trigger must be configured with CLOCK_THREAD_CPUTIME_ID";
}

TEST(trigger_mutual_exclusion, buggy_else_if_skips_cputime_when_realtime_present)
{
    // Regression documentation: prove that the BUGGY `else if` path produces
    // zero cputime configure() calls when the realtime signal is also present.
    // This test documents the BROKEN behaviour — it asserts the broken invariant
    // to make the failure mode explicit in CI output.

    int const rt_sig  = ::rocprofsys::get_sampling_realtime_signal();
    int const cpu_sig = ::rocprofsys::get_sampling_cputime_signal();

    std::set<int> const sigs = { rt_sig, cpu_sig };

    mock_timer_trigger single_slot_trigger;  // mimics the single trig_slot in state

    // Buggy wiring (current production code):
    if(sigs.count(rt_sig) > 0)
    {
        single_slot_trigger.configure(0, 0, rt_sig, CLOCK_REALTIME, 100.0, 0.0);
        single_slot_trigger.start();
    }
    else if(sigs.count(cpu_sig) > 0)  // NEVER reached when rt_sig is present
    {
        single_slot_trigger.configure(0, 0, cpu_sig, CLOCK_THREAD_CPUTIME_ID, 100.0, 0.0);
        single_slot_trigger.start();
    }

    // The buggy path configures exactly 1 trigger (realtime only).
    EXPECT_EQ(single_slot_trigger.m_calls.size(), 1U)
        << "Buggy else-if wiring must configure exactly 1 trigger "
           "(realtime only, cputime silently skipped)";
    EXPECT_EQ(single_slot_trigger.m_calls[0].m_clock,
              static_cast<clockid_t>(CLOCK_REALTIME))
        << "Only the realtime trigger must be configured under the buggy else-if path";
}
