/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Unit tests for the polling-fallback backoff helper
// (core/util/poll_backoff.h) used by Runtime::AsyncEventsLoop when the thunk
// exposes no interrupt-backed signal events (e.g. the WSL/dxg thunk). The
// helper governs how the userspace polling nap escalates, which is what keeps
// an idle async-events thread from monopolizing a CPU core
// (https://github.com/ROCm/librocdxg/issues/60).

#include <algorithm>

#include "gtest/gtest.h"

#include "core/util/poll_backoff.h"

using rocr::core::kPollNapCeilingUs;
using rocr::core::kPollNapFloorUs;
using rocr::core::NextPollNapUs;

// The documented bounds must be sane: a positive floor below the ceiling.
TEST(PollBackoffTest, BoundsAreOrdered) {
  EXPECT_GT(kPollNapFloorUs, 0);
  EXPECT_GT(kPollNapCeilingUs, kPollNapFloorUs);
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

// AsyncEventsLoop resets the nap to the floor on every new wait batch (via
// block scope). Emulate that reset and confirm a fresh wait starts cheap again
// regardless of how far a previous idle wait had escalated -- the escalated
// value never carries across waits.
TEST(PollBackoffTest, ResetReturnsToFloor) {
  int nap = kPollNapFloorUs;
  for (int i = 0; i < 10; ++i) nap = NextPollNapUs(nap);
  EXPECT_EQ(nap, kPollNapCeilingUs);

  nap = kPollNapFloorUs;  // new wait begins
  EXPECT_EQ(NextPollNapUs(nap), 2 * kPollNapFloorUs);
}
