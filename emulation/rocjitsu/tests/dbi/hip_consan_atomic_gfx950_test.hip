// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_atomic_gfx950_test.hip
/// @brief Live gfx950 proof for ConSan's 64-bit inline-shadow exchange.

#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    const hipError_t err = (call);                                                                 \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

__device__ uint64_t gfx950_flat_atomic_swap_u64(uint64_t *address, uint64_t value) {
  uint64_t old = 0;
  asm volatile("flat_atomic_swap_x2 %0, %1, %2 sc0\n\t"
               "s_waitcnt vmcnt(0) lgkmcnt(0)"
               : "=v"(old)
               : "v"(address), "v"(value)
               : "memory");
  return old;
}

__global__ void gfx950_atomic_swap_single(uint64_t *slot, uint64_t *returned,
                                          uint64_t replacement) {
  if (threadIdx.x == 0)
    returned[0] = gfx950_flat_atomic_swap_u64(slot, replacement);
}

__global__ __launch_bounds__(64) void gfx950_atomic_swap_contended(uint64_t *slot,
                                                                   uint64_t *returned) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint64_t replacement = 0x9500000000000000ull | static_cast<uint64_t>(tid + 1u);
  returned[tid] = gfx950_flat_atomic_swap_u64(slot, replacement);
}

} // namespace

TEST(ConSanAtomicGfx950Test, ReturnsOldValueAndPublishesReplacement) {
  constexpr uint64_t kInitial = 0x123456789abcdef0ull;
  constexpr uint64_t kReplacement = 0xfedcba9876543210ull;
  uint64_t *device_slot = nullptr;
  uint64_t *device_returned = nullptr;
  HIP_ASSERT(hipMalloc(&device_slot, sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc(&device_returned, sizeof(uint64_t)));
  HIP_ASSERT(hipMemcpy(device_slot, &kInitial, sizeof(kInitial), hipMemcpyHostToDevice));

  gfx950_atomic_swap_single<<<1, 1>>>(device_slot, device_returned, kReplacement);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint64_t slot = 0;
  uint64_t returned = 0;
  HIP_ASSERT(hipMemcpy(&slot, device_slot, sizeof(slot), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(&returned, device_returned, sizeof(returned), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_returned));
  HIP_ASSERT(hipFree(device_slot));
  EXPECT_EQ(returned, kInitial);
  EXPECT_EQ(slot, kReplacement);
}

TEST(ConSanAtomicGfx950Test, ContentionFormsOneAtomicExchangeChain) {
  constexpr uint32_t kThreads = 64;
  constexpr uint64_t kInitial = 0x123456789abcdef0ull;
  uint64_t *device_slot = nullptr;
  uint64_t *device_returned = nullptr;
  HIP_ASSERT(hipMalloc(&device_slot, sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc(&device_returned, kThreads * sizeof(uint64_t)));
  HIP_ASSERT(hipMemcpy(device_slot, &kInitial, sizeof(kInitial), hipMemcpyHostToDevice));

  gfx950_atomic_swap_contended<<<1, kThreads>>>(device_slot, device_returned);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint64_t slot = 0;
  std::vector<uint64_t> returned(kThreads);
  HIP_ASSERT(hipMemcpy(&slot, device_slot, sizeof(slot), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(returned.data(), device_returned, returned.size() * sizeof(uint64_t),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_returned));
  HIP_ASSERT(hipFree(device_slot));

  std::vector<uint64_t> observed = returned;
  observed.push_back(slot);
  std::ranges::sort(observed);
  std::vector<uint64_t> expected = {kInitial};
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    expected.push_back(0x9500000000000000ull | static_cast<uint64_t>(tid + 1u));
  std::ranges::sort(expected);
  EXPECT_EQ(observed, expected);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return result;
}
