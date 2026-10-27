/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Context Destroy
 * @{
 * @ingroup HRRTest
 * hipCtxDestroy must not release a context replay only observed. No GPU: the
 * refusal returns before hipCtxDestroy.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <string>

hipError_t playback_hipCtxDestroy(PlaybackContext& ctx, const uint8_t* pl);

/**
 * Test Description
 * ----------------
 *   - The map holds a context recorded by a query, with no create retain.
 *     Destroy reports that and leaves the map entry in place. It does not
 *     call the runtime.
 */
TEST_CASE("Unit_HRR_CtxDestroy_QueriedContextIsNotReleased", "[hrr]") {
  PlaybackContext ctx;
  constexpr uint64_t rec = 0x99;
  auto* sentinel = reinterpret_cast<hipCtx_t>(static_cast<uintptr_t>(0x1));
  ctx.record_ctx(rec, sentinel);

  hrr_test::ExactRecord<hrr_args_hipCtxDestroy> record;
  record.get()->ctx = rec;

  hipError_t r = hipSuccess;
  const std::string err = hrr_test::capture_stderr(
      [&] { r = playback_hipCtxDestroy(ctx, record.bytes()); });
  CHECK(r == hipErrorInvalidContext);
  CHECK(err.find("not created by replay") != std::string::npos);
  CHECK(ctx.translate_ctx(rec) == sentinel);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
