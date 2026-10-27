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
 * host_free_mu shared from before it hands the turn on (hrr_launch_hold) until
 * its launch call returns; a free takes it exclusive. No device work: the
 * runtime call of the free fails, as the pointers are not the runtime's.
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
#include <thread>

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipHostFree(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipFreeHost(PlaybackContext& ctx, const uint8_t* pl);

namespace {

constexpr uint64_t kRec = 0x77;

// Runs a launch against a free of the same buffer, as two replay threads do.
// This thread is the launch: it takes the hold, hands the turn on by moving
// next_seq as dispatch_event does, and keeps the hold while it reads the
// buffer, which stands in for resolving the destination and queuing the copy.
// `free_call` runs on the thread of the next sequence number. It must not
// return before the launch releases, and must not see the buffer change.
void launch_against_free(PlaybackContext& ctx, const unsigned char* buf,
                         const std::function<void()>& free_call) {
  std::atomic<bool> started{false};
  std::atomic<bool> freed{false};
  std::atomic<bool> released{false};
  std::atomic<bool> freed_after_release{false};

  hrr_launch_hold(ctx);
  ctx.next_seq.store(1, std::memory_order_release);

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
  CHECK(buf[0] == 0xAB);

  released.store(true);
  hrr_launch_release();
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

  launch_against_free(ctx, static_cast<unsigned char*>(b.p), [&] {
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
    launch_against_free(ctx, static_cast<unsigned char*>(b.p), [&] {
      (void)hrr_test::capture_stderr([&] { (void)playback_hipHostFree(ctx, record.bytes()); });
    });
  }
  SECTION("hipFreeHost") {
    hrr_test::ExactRecord<hrr_args_hipFreeHost> record;
    record.get()->ptr = kRec;
    (void)hrr_test::capture_stderr([&] { (void)playback_hipFreeHost(ctx, record.bytes()); });
    launch_against_free(ctx, static_cast<unsigned char*>(b.p), [&] {
      (void)hrr_test::capture_stderr([&] { (void)playback_hipFreeHost(ctx, record.bytes()); });
    });
  }
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
