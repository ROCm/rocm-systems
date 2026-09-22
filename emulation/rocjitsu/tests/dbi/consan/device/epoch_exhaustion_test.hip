// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_device_test_support.h"

namespace {

__global__ void epoch_exhaustion_for_consan(uint32_t *output, uint32_t barriers, bool ordered,
                                            uint32_t trailing_barriers) {
  __shared__ uint32_t word;
  // Runtime count prevents loop unrolling and exercises repeated dynamic barriers.
  for (uint32_t i = 0; i < barriers; ++i)
    __syncthreads();
  if (threadIdx.x == 0)
    consan_device_test::store_lds_word_without_cave(&word, 42);
  if (ordered)
    __syncthreads();
  // Exactly one lane per wave accesses the cell, avoiding intra-wave collisions.
  if (threadIdx.x % warpSize == 0)
    output[threadIdx.x / warpSize] = consan_device_test::load_lds_word_without_cave(&word);
  // Exhausting the epoch later must not erase already published race evidence.
  for (uint32_t i = 0; i < trailing_barriers; ++i)
    __syncthreads();
}

void run_epoch_boundary(uint32_t barriers, bool ordered, uint32_t trailing_barriers = 0) {
  uint32_t *output = nullptr;
  HIP_ASSERT(hipMalloc(&output, 4 * sizeof(uint32_t)));
  epoch_exhaustion_for_consan<<<1, 128>>>(output, barriers, ordered, trailing_barriers);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  if (ordered) {
    uint32_t host[4]{};
    HIP_ASSERT(hipMemcpy(host, output, sizeof(host), hipMemcpyDeviceToHost));
    for (uint32_t wave = 0; wave < 128u / consan_device_test::active_wave_size(); ++wave)
      EXPECT_EQ(host[wave], 42u);
  }
  HIP_ASSERT(hipFree(output));
}

TEST(ConSanDeviceEpochTest, LastOrderedPair) { run_epoch_boundary(1022, true); }
TEST(ConSanDeviceEpochTest, LastEpochRace) { run_epoch_boundary(1023, false); }
TEST(ConSanDeviceEpochTest, LastEpochRaceThenExhaustion) { run_epoch_boundary(1023, false, 2); }
TEST(ConSanDeviceEpochTest, ExhaustedOrderedPair) { run_epoch_boundary(1023, true); }
TEST(ConSanDeviceEpochTest, RepeatedExhaustion) { run_epoch_boundary(1100, true); }

} // namespace
