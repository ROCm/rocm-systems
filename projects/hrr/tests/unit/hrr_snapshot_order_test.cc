/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Snapshot Ordering
 * @{
 * @ingroup HRRTest
 * A host free waits until a thread that holds the snapshot lock releases it,
 * so the free cannot drop the buffer out from under a copy that is still
 * being queued. The pointer is not registered, so the runtime call fails and
 * then playback frees the buffer. No device work.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl);

/**
 * Test Description
 * ----------------
 *   - One thread holds host_free_mu shared, as the snapshot queue does, and
 *     reads the buffer. The unregister runs on another thread. It must not
 *     free the buffer until the shared lock is released.
 */
TEST_CASE("Unit_HRR_SnapshotOrder_FreeWaitsForSnapshotLock", "[hrr]") {
  void* buf = nullptr;
  REQUIRE(posix_memalign(&buf, 64, 256) == 0);
  std::memset(buf, 0xAB, 256);

  PlaybackContext ctx;
  constexpr uint64_t rec = 0x77;
  ctx.record_alloc(rec, buf, 256, AllocKind::HostRegister);
  ctx.host_reg_bufs[rec] = buf;

  std::atomic<int> phase{0};
  std::atomic<int> intact{0};
  std::thread holder([&] {
    std::shared_lock lk(ctx.host_free_mu);
    phase.store(1);
    while (phase.load() < 2) std::this_thread::yield();
    intact.store(static_cast<unsigned char*>(buf)[0] == 0xAB ? 1 : 0);
    phase.store(3);
  });
  while (phase.load() < 1) std::this_thread::yield();

  std::thread unreg([&] {
    hrr_test::ExactRecord<hrr_args_hipHostUnregister> record;
    record.get()->hostPtr = rec;
    (void)playback_hipHostUnregister(ctx, record.bytes());
    phase.store(4);
  });

  while (phase.load() < 3) {
    CHECK(phase.load() != 4);
    if (phase.load() == 1) phase.store(2);
    std::this_thread::yield();
  }
  holder.join();
  unreg.join();
  CHECK(intact.load() == 1);
  CHECK(phase.load() == 4);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
