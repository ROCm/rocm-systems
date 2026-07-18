/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// hipDeviceReset discards the current device's state - all streams, memory
// allocations, events, and device flags - and returns the device to a fresh
// state. That teardown is process-global for the device, so unlike every other
// contract domain this one is intentionally a SINGLE test case in its own
// executable: a domain binary with one case makes a direct binary run identical
// to the ctest one-process-per-test model, so the reset cannot disturb any
// sibling case sharing the process. Do not add further cases to this file.
namespace {
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}

// The schedule subfield of the device flags must be one of the documented
// scheduling modes. The exact default differs by backend (the AMD runtime resets
// to spin, CUDA to auto), so only membership in the documented set is asserted.
bool IsKnownScheduleFlag(unsigned int flags) {
  const unsigned int schedule = flags & hipDeviceScheduleMask;
  return schedule == hipDeviceScheduleAuto || schedule == hipDeviceScheduleSpin ||
         schedule == hipDeviceScheduleYield || schedule == hipDeviceScheduleBlockingSync;
}
}  // namespace

HIP_TEST_CASE(Contract_DeviceReset_DiscardsStateAndLeavesDeviceUsable) {
  RequireDevice();

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  // Allocate a buffer that the reset is expected to invalidate. Ownership stays
  // in this test case; because this is the only case in the process there is no
  // sibling that could hold a handle across the reset.
  void* stale_ptr = nullptr;
  HIP_CHECK(hipMalloc(&stale_ptr, 256));
  REQUIRE(stale_ptr != nullptr);

  // Resetting the device must succeed and return it to a fresh state.
  HIP_CHECK(hipDeviceReset());

  // The device flags are queryable after the reset and report a documented
  // scheduling mode (the runtime re-established a fresh device state).
  unsigned int flags = 0;
  HIP_CHECK(hipGetDeviceFlags(&flags));
  REQUIRE(IsKnownScheduleFlag(flags));

  // The allocation made before the reset is discarded: freeing it must not
  // silently succeed as if it were still a live allocation. The reset already
  // reclaimed the memory, so any non-success status satisfies the contract. The
  // sticky error left by the rejected free is cleared so it does not leak.
  const hipError_t stale_free = hipFree(stale_ptr);
  REQUIRE(stale_free != hipSuccess);
  (void)hipGetLastError();

  // The device must remain usable after the reset: a fresh allocation, a memset,
  // and a device-to-host copy must round-trip, and the fresh allocation frees
  // cleanly. This proves the runtime re-established a working device context.
  int* fresh_ptr = nullptr;
  HIP_CHECK(hipMalloc(&fresh_ptr, sizeof(int)));
  REQUIRE(fresh_ptr != nullptr);
  HIP_CHECK(hipMemset(fresh_ptr, 0, sizeof(int)));

  int host_value = -1;
  HIP_CHECK(hipMemcpy(&host_value, fresh_ptr, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host_value == 0);

  HIP_CHECK(hipFree(fresh_ptr));
}
