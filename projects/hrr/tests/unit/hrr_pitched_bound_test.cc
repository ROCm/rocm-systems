/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Pitched Copy Bound
 * @{
 * @ingroup HRRTest
 * A pitched copy through an interior pointer must be refused before HIP when
 * the rectangle runs past the allocation replay made. No GPU: the refusal
 * returns before the runtime call.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <string>
#include <vector>

hipError_t playback_hipMemcpy2D(PlaybackContext& ctx, const uint8_t* pl);

/**
 * Test Description
 * ----------------
 *   - A 4096-byte allocation. The destination is 2048 bytes in, with pitch
 *     2048 and height 4, so the rectangle needs 6160 bytes and 2048 remain.
 *     Replay refuses the copy and does not call HIP.
 */
TEST_CASE("Unit_HRR_PitchedCopy_InteriorPointerRefused", "[hrr]") {
  std::vector<uint8_t> buf(4096, 0x11);
  PlaybackContext ctx;
  constexpr uint64_t base = 0x1000;
  ctx.record_alloc(base, buf.data(), buf.size(), AllocKind::HostRegister);

  hrr_test::ExactRecord<hrr_args_hipMemcpy2D> record;
  auto* a = record.get();
  a->src = base;
  a->dst = base + 2048;
  a->spitch = 16;
  a->dpitch = 2048;
  a->width = 16;
  a->height = 4;
  a->kind = hipMemcpyDeviceToDevice;

  hipError_t r = hipSuccess;
  const std::string err = hrr_test::capture_stderr(
      [&] { r = playback_hipMemcpy2D(ctx, record.bytes()); });
  CHECK(r == hipErrorInvalidValue);
  CHECK(err.find("does not fit the translated allocation") != std::string::npos);
  CHECK(buf[0] == 0x11);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
