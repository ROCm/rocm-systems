/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Query Retry
 * @{
 * @ingroup HRRTest
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr_handler_test_util.h"

#include <string>

namespace {

struct Probe {
  int calls = 0;
  int succeed_on = 0;  // 0: never
  hipError_t other = hipErrorNotReady;
};

hipError_t query_probe(void* p) {
  auto* q = static_cast<Probe*>(p);
  ++q->calls;
  if (q->succeed_on && q->calls == q->succeed_on) return hipSuccess;
  return q->other;
}

}  // namespace

TEST_CASE("Unit_HRR_Query_NeverCompletingQueryEndsTheReplay", "[hrr]") {
  PlaybackContext ctx;
  ctx.max_query_attempts = 7;
  Probe probe;
  hipError_t r = hipSuccess;
  const std::string err = hrr_test::capture_stderr(
      [&] { r = replay_query_until_success(ctx, "hipStreamQuery", query_probe, &probe); });
  CHECK(r == hipErrorNotReady);
  CHECK(probe.calls == 7);
  CHECK(ctx.fatal_error.load());
  CHECK(err.find("hipStreamQuery") != std::string::npos);
  CHECK(err.find("7 attempts") != std::string::npos);
}

TEST_CASE("Unit_HRR_Query_CompletingQueryReturnsSuccess", "[hrr]") {
  PlaybackContext ctx;
  ctx.max_query_attempts = 100;
  Probe probe;
  probe.succeed_on = 3;
  CHECK(replay_query_until_success(ctx, "hipEventQuery", query_probe, &probe) == hipSuccess);
  CHECK(probe.calls == 3);
  CHECK_FALSE(ctx.fatal_error.load());
}

TEST_CASE("Unit_HRR_Query_OtherErrorIsReturnedAtOnce", "[hrr]") {
  PlaybackContext ctx;
  ctx.max_query_attempts = 100;
  Probe probe;
  probe.other = hipErrorInvalidValue;
  CHECK(replay_query_until_success(ctx, "hipEventQuery", query_probe, &probe) ==
        hipErrorInvalidValue);
  CHECK(probe.calls == 1);
  CHECK_FALSE(ctx.fatal_error.load());
}

TEST_CASE("Unit_HRR_Query_ZeroBoundMakesNoCall", "[hrr]") {
  PlaybackContext ctx;
  ctx.max_query_attempts = 0;
  Probe probe;
  hipError_t r = hipSuccess;
  (void)hrr_test::capture_stderr(
      [&] { r = replay_query_until_success(ctx, "hipEventQuery", query_probe, &probe); });
  CHECK(r == hipErrorNotReady);
  CHECK(probe.calls == 0);
  CHECK(ctx.fatal_error.load());
}

TEST_CASE("Unit_HRR_Query_DefaultHasNoAttemptBound", "[hrr]") {
  PlaybackContext ctx;
  CHECK(ctx.max_query_attempts == UINT64_MAX);
  CHECK(hrr::kDefaultMaxQueryAttempts == UINT64_MAX);
  Probe probe;
  probe.succeed_on = 1000;
  CHECK(replay_query_until_success(ctx, "hipEventQuery", query_probe, &probe) == hipSuccess);
  CHECK(probe.calls == 1000);
  CHECK_FALSE(ctx.fatal_error.load());
}

TEST_CASE("Unit_HRR_Query_StopsWhenReplayIsFatal", "[hrr]") {
  PlaybackContext ctx;
  ctx.fatal_error.store(true);
  Probe probe;
  CHECK(replay_query_until_success(ctx, "hipEventQuery", query_probe, &probe) ==
        hipErrorUnknown);
  CHECK(probe.calls == 0);
}

/** @} */
