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

#include <hip_test_common.hh>
#include "hip_playback.h"

// ROCM-27985: the post-H2D-restore drain (hrr_sync_after_replayed_h2d) must
// be skipped while a stream graph is being captured, because a device/stream
// synchronize is illegal mid-capture (HIP 900/901). Lock that guard down: a
// regression that dropped it would break HRR graph replay.
HIP_TEST_CASE(Unit_HRR_Playback_ReplayedH2DDrainGraphGuard) {
  // Outside graph capture: the replayed H2D restore must be drained.
  REQUIRE(hrr_replayed_h2d_needs_drain(false));
  // During graph capture: draining is illegal, so it must be skipped.
  REQUIRE_FALSE(hrr_replayed_h2d_needs_drain(true));
}

/**
 * End doxygen group HRR.
 * @}
 */
