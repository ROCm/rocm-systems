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
 * release. The pointer is not registered, so the runtime call fails with
 * hipErrorHostMemoryNotRegistered; without a GPU it fails with hipErrorNoDevice
 * instead, so the cases skip there.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipHostMalloc(PlaybackContext& ctx, const uint8_t* pl);

namespace {
bool have_device() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
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
  if (!have_device()) SKIP("needs a GPU: without one the call fails with hipErrorNoDevice");

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
 *   - A second hipHostMalloc replays onto an address that still holds a
 *     registration the runtime refuses to release. The old buffer is not
 *     freed and keeps its charge against the host budget.
 */
TEST_CASE("Unit_HRR_HostUnregister_ReplacedRegistrationKeepsBuffer", "[hrr]") {
  if (!have_device()) SKIP("needs a GPU: without one the call fails with hipErrorNoDevice");

  void* buf = nullptr;
  REQUIRE(posix_memalign(&buf, 64, 256) == 0);
  std::memset(buf, 0x5A, 256);

  PlaybackContext ctx;
  constexpr uint64_t rec = 0x42;
  REQUIRE(ctx.host_budget.try_charge(256));
  ctx.record_alloc(rec, buf, 256, AllocKind::HostRegister);
  ctx.host_reg_bufs[rec] = buf;

  hrr_test::ExactRecord<hrr_args_hipHostMalloc> record;
  record.get()->ptr = rec;
  record.get()->size = 64;

  hipError_t r = hipErrorUnknown;
  (void)hrr_test::capture_stderr([&] { r = playback_hipHostMalloc(ctx, record.bytes()); });
  REQUIRE(r == hipSuccess);
  CHECK(ctx.host_budget.used() == 256 + 64);
  // Under AddressSanitizer a freed buffer fails this read.
  CHECK(static_cast<unsigned char*>(buf)[0] == 0x5A);

  (void)hipHostFree(ctx.lookup_ptr(rec));
  free(buf);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
