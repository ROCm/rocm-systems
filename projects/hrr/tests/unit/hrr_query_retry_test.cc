/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Query Retry
 * @{
 * @ingroup HRRTest
 * Handler-level tests (hrr-handler-tests links hip_playback.cpp in) for the
 * bounded hipEventQuery/hipStreamQuery retry. They call
 * replay_query_until_success with stand-ins for the HIP query, so no HIP call
 * is made and no GPU is needed.
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

/**
 * Test Description
 * ----------------
 *   - A query that never completes is called exactly max_query_attempts times,
 *     then the replay ends: the result is hipErrorNotReady, the message names
 *     the API and the bound, and fatal_error is set (so --continue-on-error
 *     does not carry on past it).
 */
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

/**
 * Test Description
 * ----------------
 *   - A query that completes before the bound returns hipSuccess after
 *     exactly the calls it took, and the replay is not ended.
 */
TEST_CASE("Unit_HRR_Query_CompletingQueryReturnsSuccess", "[hrr]") {
  PlaybackContext ctx;
  ctx.max_query_attempts = 100;
  Probe probe;
  probe.succeed_on = 3;
  CHECK(replay_query_until_success(ctx, "hipEventQuery", query_probe, &probe) == hipSuccess);
  CHECK(probe.calls == 3);
  CHECK_FALSE(ctx.fatal_error.load());
}

/**
 * Test Description
 * ----------------
 *   - An error other than hipErrorNotReady is returned at once, without
 *     retrying and without ending the replay on its own.
 */
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

/**
 * Test Description
 * ----------------
 *   - A bound of zero makes no query at all and ends the replay.
 */
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

/** @} */
