/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Host Unregister
 * @{
 * @ingroup HRRTest
 * hipHostUnregister must not free a replay buffer: not when the call fails, and
 * not when it succeeds while a graph node still points into the buffer. A
 * buffer whose unregister failed is not reused. The cases call HIP, so they
 * skip without a GPU.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

hipError_t playback_hipHostRegister(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl);

namespace {
// Replays a hipHostRegister of size bytes at the recorded address rec. The blob
// is not in the archive, so the buffer is zero-filled.
hipError_t replay_register(PlaybackContext& ctx, uint64_t rec, size_t size) {
  hrr_test::ExactRecord<hrr_args_hipHostRegister> reg;
  reg.get()->hostPtr = rec;
  reg.get()->sizeBytes = size;
  reg.get()->blob_hash_lo = 1;
  hipError_t r = hipErrorUnknown;
  (void)hrr_test::capture_stderr([&] { r = playback_hipHostRegister(ctx, reg.bytes()); });
  return r;
}
}  // namespace

/**
 * Test Description
 * ----------------
 *   - The recorded registration names a buffer this process allocated and
 *     never registered. hipHostUnregister fails. The buffer and both maps
 *     must still name it afterwards, so a later snapshot can write it.
 */
TEST_CASE("Unit_HRR_HostUnregister_FailedCallKeepsBuffer", "[hrr]") {
  if (!hrr_test::have_device()) SKIP("needs a GPU: without one the call fails with hipErrorNoDevice");

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
  CHECK(r == hipErrorHostMemoryNotRegistered);
  CHECK(ctx.host_reg_bufs.count(rec) == 1);
  CHECK(ctx.lookup_ptr(rec) == buf);
  // Under AddressSanitizer a freed buffer fails this read.
  CHECK(static_cast<unsigned char*>(buf)[0] == 0x5A);

  free(buf);
}

/**
 * Test Description
 * ----------------
 *   - The address is registered again and unregistering the old block fails
 *     (it was unregistered behind replay's back). The new registration gets a
 *     fresh block rather than the old one.
 */
TEST_CASE("Unit_HRR_HostUnregister_FailedReplaceTakesNewBlock", "[hrr]") {
  if (!hrr_test::have_device()) SKIP("needs a GPU to register host memory");

  PlaybackContext ctx;
  REQUIRE(replay_register(ctx, 0x1000, 256) == hipSuccess);
  void* old = ctx.lookup_ptr(0x1000);
  REQUIRE(hipHostUnregister(old) == hipSuccess);

  REQUIRE(replay_register(ctx, 0x1000, 256) == hipSuccess);
  void* fresh = ctx.lookup_ptr(0x1000);
  CHECK(fresh != old);
  CHECK(hipHostUnregister(fresh) == hipSuccess);
}

/**
 * Test Description
 * ----------------
 *   - A graph memcpy node built on a registered address keeps the buffer
 *     pointer replay translated it to. The address is registered again, which
 *     reuses the buffer, then unregistered. The buffer must stay writable,
 *     charged once, until the context is destroyed. Under AddressSanitizer a
 *     freed buffer fails the write.
 */
TEST_CASE("Unit_HRR_HostUnregister_GraphNodeKeepsBuffer", "[hrr]") {
  if (!hrr_test::have_device()) SKIP("needs a GPU to register host memory");

  constexpr uint64_t kHost = 0x1000;
  constexpr size_t kSize = 256;
  PlaybackContext ctx;
  REQUIRE(replay_register(ctx, kHost, kSize) == hipSuccess);
  // What a hipGraphAddMemcpyNode1D on kHost is given (translate_ptr).
  auto* node_ptr = static_cast<unsigned char*>(ctx.translate_ptr(kHost));
  REQUIRE(node_ptr != nullptr);

  REQUIRE(replay_register(ctx, kHost, kSize) == hipSuccess);
  CHECK(ctx.lookup_ptr(kHost) == node_ptr);

  hrr_test::ExactRecord<hrr_args_hipHostUnregister> unreg;
  unreg.get()->hostPtr = kHost;
  REQUIRE(playback_hipHostUnregister(ctx, unreg.bytes()) == hipSuccess);
  CHECK_FALSE(ctx.has_alloc(kHost));
  CHECK(ctx.host_budget.used() == kSize);

  // What the node's next launch does to its destination.
  std::memset(node_ptr, 0x5A, kSize);
  CHECK(node_ptr[kSize - 1] == 0x5A);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
