/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Snapshot Ordering
 * @{
 * @ingroup HRRTest
 * A kernel launch gives up the global replay turn before its handler runs, so
 * the free of a pinned host buffer at the next sequence number can start while
 * the launch is still queuing its snapshot of that buffer. The launch holds
 * host_free_mu shared from before it hands the turn on until its launch call
 * returns; a free takes it exclusive. dispatch_event hands the turn on through
 * HrrTurnGuard, which takes the hold first for a launch, and the tests drive
 * that guard. They do not run dispatch_event: that it passes the guard the
 * right launch flag is not covered, only that hrr_is_launch_handler names the
 * launch handlers and not the frees. No device work: the runtime call of the
 * free fails, as the pointers are not the runtime's, so what the tests observe
 * is whether the free call returns while the hold is taken.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipHostFree(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipFreeHost(PlaybackContext& ctx, const uint8_t* pl);

namespace {

constexpr uint64_t kRec = 0x77;

// Runs a launch against a free of the same buffer, as two replay threads do.
// This thread is the launch: it hands the turn on through the HrrTurnGuard of
// a launch event, as dispatch_event does, which takes the hold and moves
// next_seq to 1, and keeps the guard while the launch would resolve the
// destination and queue the copy. `free_call` runs on the thread of the next
// sequence number. It must not return before the guard is gone.
void launch_against_free(PlaybackContext& ctx, const std::function<void()>& free_call) {
  std::atomic<bool> started{false};
  std::atomic<bool> freed{false};
  std::atomic<bool> released{false};
  std::atomic<bool> freed_after_release{false};

  auto turn = std::make_unique<HrrTurnGuard>(ctx, 1, /*ordered=*/false, /*launch=*/true);

  std::thread next_turn([&] {
    while (ctx.next_seq.load(std::memory_order_acquire) < 1) std::this_thread::yield();
    started.store(true);
    free_call();
    freed_after_release.store(released.load());
    freed.store(true);
  });
  while (!started.load()) std::this_thread::yield();

  // Give a free that is not held back every chance to finish.
  for (int i = 0; i < 30 && !freed.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK_FALSE(freed.load());

  released.store(true);
  turn.reset();
  next_turn.join();
  CHECK(freed_after_release.load());
}

// A pinned buffer the replay owns, recorded as `rec`.
struct Buffer {
  void* p = nullptr;
  Buffer() {
    REQUIRE(posix_memalign(&p, 64, 256) == 0);
    std::memset(p, 0xAB, 256);
  }
  ~Buffer() { free(p); }
};

}  // namespace

/**
 * Test Description
 * ----------------
 *   - The memory came from hipHostRegister: hipHostUnregister, which the
 *     runtime refuses here, must wait for a launch that holds the lock.
 */
TEST_CASE("Unit_HRR_SnapshotOrder_UnregisterWaitsForLaunch", "[hrr]") {
  Buffer b;
  PlaybackContext ctx;
  ctx.record_alloc(kRec, b.p, 256, AllocKind::HostRegister);
  ctx.host_reg_bufs[kRec] = b.p;

  hrr_test::ExactRecord<hrr_args_hipHostUnregister> record;
  record.get()->hostPtr = kRec;
  // The first runtime call can take long; keep that out of the timed window.
  (void)hrr_test::capture_stderr([&] { (void)playback_hipHostUnregister(ctx, record.bytes()); });

  launch_against_free(ctx, [&] {
    (void)hrr_test::capture_stderr([&] { (void)playback_hipHostUnregister(ctx, record.bytes()); });
  });
}

/**
 * Test Description
 * ----------------
 *   - The memory came from hipHostMalloc: hipHostFree and hipFreeHost must
 *     wait for a launch that holds the lock.
 */
TEST_CASE("Unit_HRR_SnapshotOrder_HostFreeWaitsForLaunch", "[hrr]") {
  Buffer b;
  PlaybackContext ctx;
  ctx.record_alloc(kRec, b.p, 256, AllocKind::HostMalloc);

  SECTION("hipHostFree") {
    hrr_test::ExactRecord<hrr_args_hipHostFree> record;
    record.get()->ptr = kRec;
    (void)hrr_test::capture_stderr([&] { (void)playback_hipHostFree(ctx, record.bytes()); });
    launch_against_free(ctx, [&] {
      (void)hrr_test::capture_stderr([&] { (void)playback_hipHostFree(ctx, record.bytes()); });
    });
  }
  SECTION("hipFreeHost") {
    hrr_test::ExactRecord<hrr_args_hipFreeHost> record;
    record.get()->ptr = kRec;
    (void)hrr_test::capture_stderr([&] { (void)playback_hipFreeHost(ctx, record.bytes()); });
    launch_against_free(ctx, [&] {
      (void)hrr_test::capture_stderr([&] { (void)playback_hipFreeHost(ctx, record.bytes()); });
    });
  }
}

/**
 * Test Description
 * ----------------
 *   - The handlers a launch event dispatches to are the ones dispatch_event
 *     gives the launch hold, and the frees are not.
 */
TEST_CASE("Unit_HRR_SnapshotOrder_LaunchHandlersTakeTheHold", "[hrr]") {
  CHECK(hrr_is_launch_handler(hrr_playback_dispatch[HRR_API_HIPMODULELAUNCHKERNEL]));
  CHECK(hrr_is_launch_handler(hrr_playback_dispatch[HRR_API_HIPLAUNCHKERNEL]));
  CHECK_FALSE(hrr_is_launch_handler(hrr_playback_dispatch[HRR_API_HIPHOSTFREE]));
  CHECK_FALSE(hrr_is_launch_handler(hrr_playback_dispatch[HRR_API_HIPFREEHOST]));
  CHECK_FALSE(hrr_is_launch_handler(hrr_playback_dispatch[HRR_API_HIPHOSTUNREGISTER]));
}

/**
 * Test Description
 * ----------------
 *   - A launch's guard takes the hold before it moves next_seq, and neither
 *     the handler's release nor the guard's end leaves the lock held.
 */
TEST_CASE("Unit_HRR_SnapshotOrder_GuardReleasesTheHold", "[hrr]") {
  PlaybackContext ctx;
  {
    HrrTurnGuard turn(ctx, 1, /*ordered=*/false, /*launch=*/true);
    CHECK(ctx.next_seq.load() == 1);
    CHECK_FALSE(ctx.host_free_mu.try_lock());
    hrr_launch_release();  // the launch handler, once its launch call returned
    CHECK(ctx.host_free_mu.try_lock());
    ctx.host_free_mu.unlock();
  }
  CHECK(ctx.host_free_mu.try_lock());
  ctx.host_free_mu.unlock();
}

/**
 * Test Description
 * ----------------
 *   - hipHostUnregister fails, so the registration still stands. The buffer
 *     and both map entries must survive it: a snapshot queued to the buffer,
 *     or the next launch's lookup, would otherwise reach freed heap, which
 *     AddressSanitizer reports.
 */
TEST_CASE("Unit_HRR_SnapshotOrder_FailedUnregisterKeepsBuffer", "[hrr]") {
  Buffer b;
  PlaybackContext ctx;
  ctx.record_alloc(kRec, b.p, 256, AllocKind::HostRegister);
  ctx.host_reg_bufs[kRec] = b.p;

  hrr_test::ExactRecord<hrr_args_hipHostUnregister> record;
  record.get()->hostPtr = kRec;
  hipError_t r = hipSuccess;
  (void)hrr_test::capture_stderr([&] { r = playback_hipHostUnregister(ctx, record.bytes()); });

  CHECK(r != hipSuccess);
  CHECK(ctx.host_reg_bufs.count(kRec) == 1);
  CHECK(ctx.lookup_ptr(kRec) == b.p);
  static_cast<unsigned char*>(b.p)[255] = 0x11;  // heap-use-after-free if freed
  CHECK(static_cast<unsigned char*>(b.p)[255] == 0x11);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
