/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Host Unregister
 * @{
 * @ingroup HRRTest
 * A failed hipHostUnregister must not free the replay buffer it did not
 * release. No GPU: the pointer is not registered, so the runtime call fails.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl);

/**
 * Test Description
 * ----------------
 *   - The recorded registration names a buffer this process allocated and
 *     never registered. hipHostUnregister fails. The buffer and both maps
 *     must still name it afterwards, so a later snapshot can write it.
 */
TEST_CASE("Unit_HRR_HostUnregister_FailedCallKeepsBuffer", "[hrr]") {
  void* buf = nullptr;
  REQUIRE(posix_memalign(&buf, 64, 256) == 0);
  std::memset(buf, 0x5A, 256);

  PlaybackContext ctx;
  constexpr uint64_t rec = 0x42;
  ctx.record_alloc(rec, buf, 256, AllocKind::HostRegister);
  ctx.host_reg_bufs[rec] = buf;

  hrr_test::ExactRecord<hrr_args_hipHostUnregister> record;
  record.get()->hostPtr = rec;

  hipError_t r = hipSuccess;
  (void)hrr_test::capture_stderr([&] { r = playback_hipHostUnregister(ctx, record.bytes()); });
  CHECK(r != hipSuccess);
  CHECK(ctx.host_reg_bufs.count(rec) == 1);
  CHECK(ctx.lookup_ptr(rec) == buf);
  CHECK(static_cast<unsigned char*>(buf)[0] == 0x5A);
  static_cast<unsigned char*>(buf)[255] = 0x11;
  CHECK(static_cast<unsigned char*>(buf)[255] == 0x11);

  ctx.host_reg_bufs.erase(rec);
  ctx.remove_alloc(rec);
  free(buf);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
