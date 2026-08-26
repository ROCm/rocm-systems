/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Playback
 * @{
 * @ingroup HRRTest
 * CPU-only unit tests for HRR playback helper invariants (no GPU required).
 */

#include "hrr_test_common.hh"
#include "hip_playback.h"

// ROCM-27985: the post-H2D-restore drain (hrr_sync_after_replayed_h2d) must
// be skipped while a stream graph is being captured, because a device/stream
// synchronize is illegal mid-capture (HIP 900/901). Lock that guard down: a
// regression that dropped it would break HRR graph replay.
HRR_TEST_CASE(Unit_HRR_Playback_ReplayedH2DDrainGraphGuard) {
  // Outside graph capture: the replayed H2D restore must be drained.
  REQUIRE(hrr_replayed_h2d_needs_drain(false));
  // During graph capture: draining is illegal, so it must be skipped.
  REQUIRE_FALSE(hrr_replayed_h2d_needs_drain(true));
}

// ROCM-27985: hipMalloc is host-synchronous, so the zero-init injected after a
// host-synchronous replay allocation must be drained before the handler
// returns. Without that edge the null-stream zero-init races every later
// replayed event on a hipStreamNonBlocking stream and can overwrite a restored
// kernel input with zeros. Draining is skipped when zero-init is turned off and
// during graph capture, where the zero-init is never issued.
HRR_TEST_CASE(Unit_HRR_Playback_ZeroInitDrainGuard) {
  // Normal replay: the zero-init must be ordered ahead of later events.
  REQUIRE(hrr_zero_init_needs_drain(true, false));
  // HIP_HRR_REPLAY_ZERO_INIT=0: nothing was issued, so nothing to drain.
  REQUIRE_FALSE(hrr_zero_init_needs_drain(false, false));
  // During graph capture: zero-init is skipped and a sync would be illegal.
  REQUIRE_FALSE(hrr_zero_init_needs_drain(true, true));
  REQUIRE_FALSE(hrr_zero_init_needs_drain(false, true));
}

/**
 * End doxygen group HRR.
 * @}
 */
