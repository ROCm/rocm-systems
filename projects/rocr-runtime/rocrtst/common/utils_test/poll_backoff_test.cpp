/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Unit tests for the polling-fallback backoff helper
// (core/util/poll_backoff.h). It is used in two places that cannot arm an
// interrupt-backed wait and would otherwise re-scan signal values in a loop
// that monopolizes a CPU core:
//   - Runtime::AsyncEventsLoop, when the thunk exposes no interrupt-backed
//     signal events (e.g. the WSL/dxg thunk, https://github.com/ROCm/librocdxg/issues/60);
//   - BusyWaitSignal::WaitRelaxed (default_signal.cpp), for a GPU-only signal
//     that has no event at all -- e.g. a host<->device copy completion signal
//     that goes unsatisfied for tens of seconds under memory pressure
//     (https://github.com/ROCm/TheRock/issues/7832).
// The helper governs the hot-spin window before napping and how the nap
// escalates afterwards.

#include <algorithm>

#include "gtest/gtest.h"

#include "core/util/poll_backoff.h"

using rocr::core::HotPollUs;
using rocr::core::kHotPollActiveUs;
using rocr::core::kHotPollUs;
using rocr::core::kPollNapCeilingMixedUs;
using rocr::core::kPollNapCeilingUs;
using rocr::core::kPollNapFloorUs;
using rocr::core::NextPollNapUs;

// The documented bounds must be sane: a positive floor below the mixed-batch
// ceiling, which in turn sits below the polling-only ceiling.
TEST(PollBackoffTest, BoundsAreOrdered) {
  EXPECT_GT(kPollNapFloorUs, 0);
  EXPECT_GT(kPollNapCeilingMixedUs, kPollNapFloorUs);
  EXPECT_GT(kPollNapCeilingUs, kPollNapCeilingMixedUs);
}

// Each step doubles the nap until it saturates at the ceiling.
TEST(PollBackoffTest, EscalatesByDoublingThenSaturates) {
  EXPECT_EQ(NextPollNapUs(kPollNapFloorUs), 2 * kPollNapFloorUs);

  int nap = kPollNapFloorUs;
  for (int i = 0; i < 64; ++i) {
    int next = NextPollNapUs(nap);
    EXPECT_LE(next, kPollNapCeilingUs);  // never exceeds the cap
    EXPECT_GE(next, nap);                // monotonic non-decreasing
    if (nap < kPollNapCeilingUs) {
      EXPECT_EQ(next, std::min(2 * nap, kPollNapCeilingUs));
    }
    nap = next;
  }
  EXPECT_EQ(nap, kPollNapCeilingUs);  // converged to the cap
}

// The ceiling is a fixed point: once reached the value cannot grow, so a
// long-lived idle wait stays at the cheap-CPU cap rather than overflowing.
TEST(PollBackoffTest, CeilingIsFixedPoint) {
  EXPECT_EQ(NextPollNapUs(kPollNapCeilingUs), kPollNapCeilingUs);
  EXPECT_EQ(NextPollNapUs(2 * kPollNapCeilingUs), kPollNapCeilingUs);
}

// AsyncEventsLoop uses the smaller mixed-batch ceiling when interrupts are
// available but a polling-only signal (IPC / DefaultSignal) forced the batch
// into polling: escalation must saturate at that ceiling, never above it, and
// the ceiling is likewise a fixed point.
TEST(PollBackoffTest, MixedCeilingSaturates) {
  int nap = kPollNapFloorUs;
  while (nap < kPollNapCeilingMixedUs) {
    nap = NextPollNapUs(nap, kPollNapCeilingMixedUs);
    EXPECT_LE(nap, kPollNapCeilingMixedUs);
  }
  EXPECT_EQ(nap, kPollNapCeilingMixedUs);
  EXPECT_EQ(NextPollNapUs(nap, kPollNapCeilingMixedUs), kPollNapCeilingMixedUs);
  EXPECT_EQ(NextPollNapUs(2 * kPollNapCeilingMixedUs, kPollNapCeilingMixedUs),
            kPollNapCeilingMixedUs);
}

// AsyncEventsLoop resets the nap to the floor on every new wait batch (via
// block scope). Emulate that reset and confirm a fresh wait starts cheap again
// regardless of how far a previous idle wait had escalated -- the escalated
// value never carries across waits.
TEST(PollBackoffTest, ResetReturnsToFloor) {
  int nap = kPollNapFloorUs;
  while (nap < kPollNapCeilingUs) nap = NextPollNapUs(nap);
  EXPECT_EQ(nap, kPollNapCeilingUs);

  nap = kPollNapFloorUs;  // new wait begins
  EXPECT_EQ(NextPollNapUs(nap), 2 * kPollNapFloorUs);
}

// --- Hot-poll window (BusyWaitSignal::WaitRelaxed) -----------------------------

// The hot-spin window must be a positive, bounded number of microseconds: long
// enough to absorb a prompt GPU completion without a nap, short enough that a
// stalled wait stops burning the core quickly. The active hint only widens it.
TEST(PollBackoffTest, HotPollWindowBounds) {
  EXPECT_GT(kHotPollUs, 0);
  EXPECT_GE(kHotPollActiveUs, kHotPollUs);
  // A blocked-hint waiter should give up the core in well under a scheduler
  // tick; an active-hint waiter within a few milliseconds.
  EXPECT_LE(kHotPollUs, 1000);
  EXPECT_LE(kHotPollActiveUs, 20000);
}

// HotPollUs() selects the window from the wait-state hint.
TEST(PollBackoffTest, HotPollSelectsOnActiveHint) {
  EXPECT_EQ(HotPollUs(false), kHotPollUs);
  EXPECT_EQ(HotPollUs(true), kHotPollActiveUs);
}

// The whole point: once a BusyWaitSignal wait is past its hot window, its CPU
// duty cycle collapses. Model the loop -- spin for the hot window, then
// os::uSleep(nap) with nap escalating via NextPollNapUs() -- over a wait far
// longer than any real GPU op (the TheRock#7832 stalls ran ~60s) and confirm
// the time spent spinning is a tiny fraction of the wait.
TEST(PollBackoffTest, LongWaitIsMostlyAsleep) {
  constexpr long kWaitUs = 60L * 1000 * 1000;  // 60 s

  long spinning_us = HotPollUs(false);  // the hot window, spent hot
  long elapsed_us = spinning_us;
  int nap = kPollNapFloorUs;
  long naps = 0;
  while (elapsed_us < kWaitUs) {
    elapsed_us += nap;  // asleep for this long
    nap = NextPollNapUs(nap);
    ++naps;
  }

  // Per nap the loop does O(1) work (one atomic load, one clock read) before
  // sleeping again; even at a generous 1 us of work per wake the spin cost is
  // negligible next to the wait.
  long spin_plus_wake_us = spinning_us + naps;
  EXPECT_LT(spin_plus_wake_us * 100, kWaitUs)  // < 1% duty cycle
      << "naps=" << naps << " spin_plus_wake_us=" << spin_plus_wake_us;

  // And the nap must have saturated at the ceiling rather than growing without
  // bound or staying tiny.
  EXPECT_EQ(nap, kPollNapCeilingUs);
}
