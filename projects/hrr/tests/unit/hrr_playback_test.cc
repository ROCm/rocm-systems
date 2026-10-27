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
#include "hrr_test_process.hh"
#include "hip_playback.h"
#include "hrr/hrr_api_args.h"

#include <cstdlib>
#include <string>

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

// The reader accepts only its own archive format version, so the number
// --version prints is what decides whether a given capture can be opened at
// all. Tie it to HRR_VERSION rather than to a literal, so a future format bump
// cannot leave the tool reporting a version it no longer reads.
HRR_TEST_CASE(Unit_HRR_Playback_VersionOption) {
  hrr::test::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(proc);

  REQUIRE(proc.run("--version") == 0);

  const std::string out = proc.getOutput();
  INFO("hrr-playback --version printed: " << out);
  REQUIRE(out.find("hrr-playback (HIP Record & Replay)") != std::string::npos);
  REQUIRE(out.find("archive format version : " + std::to_string(HRR_VERSION)) !=
          std::string::npos);

  // The build identity must be reported. Its value is not compared: a build
  // from a tree without git history reports "unknown" by design.
  REQUIRE(out.find("source revision        : ") != std::string::npos);
}

// Placement holds each recorded allocation rounded out to whole pages, merges
// what overlaps or touches, and leaves out an allocation exported over IPC
// and any other over the same pages.
HRR_TEST_CASE(Unit_HRR_Playback_PlacementPlan) {
  const uint64_t P = 4096, B = 0x7f0000000000ull;
  auto plan = hrr_place_plan({{B + 3 * P, B + 3 * P + 8},  // one page
                              {B, B + P + 1},              // two pages
                              {B + 2 * P, B + 3 * P},      // touches both
                              {B + 9 * P, B + 10 * P},     // exported below
                              {B + 9 * P + 64, B + 12 * P},  // reuses its pages
                              {0, 64}},                    // not an address
                             {B + 9 * P + 16});
  REQUIRE(plan.size() == 1);
  CHECK(plan[0].base == B);
  CHECK(plan[0].end == B + 4 * P);
}

// An allocation is placed only inside a range still held, and its range is
// held again once it is freed.
HRR_TEST_CASE(Unit_HRR_Playback_PlacementHeldRanges) {
  HrrHeldRanges h;
  h.add(0x10000, 0x20000);
  CHECK_FALSE(h.take(0x18000, 0x21000));  // runs past the hold
  REQUIRE(h.take(0x14000, 0x18000));
  CHECK_FALSE(h.take(0x15000, 0x16000));  // already taken
  CHECK(h.r.size() == 2);
  h.add(0x14000, 0x18000);
  REQUIRE(h.r.size() == 1);
  CHECK(h.r.begin()->first == 0x10000);
  CHECK(h.r.begin()->second == 0x20000);
}

// hrr_place_range leaves the ceiling and address 0 to the held ranges: the plan
// holds neither, so take refuses a range that reaches them (AIRUNTIME-3069).
HRR_TEST_CASE(Unit_HRR_Playback_PlacementHeldRangesBounded) {
  const uint64_t G = 1ull << 21, top = 1ull << 47;
  HrrHeldRanges h;
  for (const auto& r : hrr_place_plan({{0, 64}, {top - G, top}, {top, top + G}}, {}))
    h.add(r.base, r.end);
  REQUIRE(h.r.size() == 1);
  uint64_t b = 0, e = 0;
  REQUIRE(hrr_place_range(0, 64, G, &b, &e));
  CHECK_FALSE(h.take(b, e));
  REQUIRE(hrr_place_range(top - 64, 65, G, &b, &e));  // one byte over
  CHECK_FALSE(h.take(b, e));
  REQUIRE(hrr_place_range(top - 64, 64, G, &b, &e));
  CHECK(h.take(b, e));
}

// The range placement maps for an allocation is checked: a size that wraps
// 2^64, or rounding that does, is refused rather than turned into a small range
// inside a held one (AIRUNTIME-3069).
HRR_TEST_CASE(Unit_HRR_Playback_PlacementRangeChecked) {
  const uint64_t G = 1ull << 21, B = 0x7f0000000000ull;
  uint64_t b = 0, e = 0;
  REQUIRE(hrr_place_range(B + 5, 100, G, &b, &e));
  CHECK(b == B);
  CHECK(e == B + G);
  REQUIRE(hrr_place_range(B, G + 1, G, &b, &e));  // spills into a second granule
  CHECK(e == B + 2 * G);

  // rec + size lands exactly on 2^64, and one past it.
  CHECK_FALSE(hrr_place_range(B, 0 - B, G, &b, &e));
  CHECK_FALSE(hrr_place_range(B, 0 - B + 1, G, &b, &e));
  CHECK_FALSE(hrr_place_range(B, ~0ull, G, &b, &e));
  // Without the rec + size check this is accepted as [B, B + G).
  CHECK_FALSE(hrr_place_range(B + 5, ~0ull, G, &b, &e));
  // rec + size ends just below 2^64: no wrap, but rounding up to a granule does.
  CHECK_FALSE(hrr_place_range(B, (0 - B) - 1, G, &b, &e));
  // The same from a different start.
  CHECK_FALSE(hrr_place_range(1ull << 63, (1ull << 63) - 1, G, &b, &e));
  // Nothing to place.
  CHECK_FALSE(hrr_place_range(B, 0, G, &b, &e));
}

/**
 * End doxygen group HRR.
 * @}
 */
