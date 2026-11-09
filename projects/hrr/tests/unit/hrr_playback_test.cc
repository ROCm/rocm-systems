/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Playback
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"
#include "hip_playback.h"
#include "hrr/hrr_api_args.h"

#include <cstdlib>
#include <string>

// A device or stream synchronize is illegal during stream graph capture (HIP 900/901),
// so the post-H2D drain must be skipped then.
HRR_TEST_CASE(Unit_HRR_Playback_ReplayedH2DDrainGraphGuard) {
  REQUIRE(hrr_replayed_h2d_needs_drain(false));
  REQUIRE_FALSE(hrr_replayed_h2d_needs_drain(true));
}

// hipMalloc is host-synchronous, so the injected zero-init must drain before the handler returns.
// Otherwise it races later events on a hipStreamNonBlocking stream and can zero a restored input.
// No drain when zero-init is off or during graph capture.
HRR_TEST_CASE(Unit_HRR_Playback_ZeroInitDrainGuard) {
  REQUIRE(hrr_zero_init_needs_drain(true, false));
  REQUIRE_FALSE(hrr_zero_init_needs_drain(false, false));
  REQUIRE_FALSE(hrr_zero_init_needs_drain(true, true));
  REQUIRE_FALSE(hrr_zero_init_needs_drain(false, true));
}

// Tie the expected value to HRR_VERSION, not a literal, so a format bump
// cannot leave --version reporting a version the reader rejects.
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

HRR_TEST_CASE(Unit_HRR_Playback_PlacementRangeChecked) {
  const uint64_t G = 1ull << 21, B = 0x7f0000000000ull;
  uint64_t b = 0, e = 0;
  REQUIRE(hrr_place_range(B + 5, 100, G, &b, &e));
  CHECK(b == B);
  CHECK(e == B + G);
  REQUIRE(hrr_place_range(B, G + 1, G, &b, &e));  // spills into a second granule
  CHECK(e == B + 2 * G);

  CHECK_FALSE(hrr_place_range(B, 0 - B, G, &b, &e));
  CHECK_FALSE(hrr_place_range(B, 0 - B + 1, G, &b, &e));
  CHECK_FALSE(hrr_place_range(B, ~0ull, G, &b, &e));
  // Without the rec + size check this is accepted as [B, B + G).
  CHECK_FALSE(hrr_place_range(B + 5, ~0ull, G, &b, &e));
  // rec + size ends just below 2^64: no wrap, but rounding up to a granule does.
  CHECK_FALSE(hrr_place_range(B, (0 - B) - 1, G, &b, &e));
  CHECK_FALSE(hrr_place_range(1ull << 63, (1ull << 63) - 1, G, &b, &e));
  CHECK_FALSE(hrr_place_range(B, 0, G, &b, &e));
}

/**
 * End doxygen group HRR.
 * @}
 */
