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
#include <hip_test_process.hh>
#include "hip_playback.h"
#include "hrr_api_args.h"

#include <cstdlib>
#include <string>

// Platform path separator for setEnv("PATH", ...): ';' on Windows, ':' on POSIX.
#ifdef _WIN32
static constexpr char kPathSep = ';';
#else
static constexpr char kPathSep = ':';
#endif

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

// ROCM-27985: hipMalloc is host-synchronous, so the zero-init injected after a
// host-synchronous replay allocation must be drained before the handler
// returns. Without that edge the null-stream zero-init races every later
// replayed event on a hipStreamNonBlocking stream and can overwrite a restored
// kernel input with zeros. Draining is skipped when zero-init is turned off and
// during graph capture, where the zero-init is never issued.
HIP_TEST_CASE(Unit_HRR_Playback_ZeroInitDrainGuard) {
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
HIP_TEST_CASE(Unit_HRR_Playback_VersionOption) {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  const char* cur_path = getenv("PATH");
  proc.setEnv("PATH",
              std::string(ROCM_BIN_PATH) + kPathSep + (cur_path ? cur_path : ""));

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

/**
 * End doxygen group HRR.
 * @}
 */
