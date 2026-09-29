// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_device_test_support.h"

namespace {

__global__ void epoch_exhaustion_for_consan(uint32_t *output, uint32_t barriers, bool ordered,
                                            uint32_t trailing_barriers, bool early_write) {
  __shared__ uint32_t word;
  if (early_write && threadIdx.x == 0)
    consan_device_test::store_lds_word_without_cave(&word, 42);
  // Runtime count prevents loop unrolling and exercises repeated dynamic barriers.
  for (uint32_t i = 0; i < barriers; ++i)
    __syncthreads();
  if (!early_write && threadIdx.x == 0)
    consan_device_test::store_lds_word_without_cave(&word, 42);
  if (ordered)
    __syncthreads();
  // Exactly one lane per wave accesses the cell, avoiding intra-wave collisions.
  if (threadIdx.x % warpSize == 0)
    output[threadIdx.x / warpSize] = consan_device_test::load_lds_word_without_cave(&word);
  // Crossing a tag boundary later must not erase already published race evidence.
  for (uint32_t i = 0; i < trailing_barriers; ++i)
    __syncthreads();
}

void run_epoch_boundary(uint32_t barriers, bool ordered, uint32_t trailing_barriers = 0,
                        bool early_write = false) {
  uint32_t *output = nullptr;
  HIP_ASSERT(hipMalloc(&output, 4 * sizeof(uint32_t)));
  epoch_exhaustion_for_consan<<<1, 128>>>(output, barriers, ordered, trailing_barriers,
                                          early_write);
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

TEST(ConSanDeviceEpochTest, BeforeTagWrapOrdered) { run_epoch_boundary(1022, true); }
TEST(ConSanDeviceEpochTest, BeforeTagWrapRace) { run_epoch_boundary(1023, false); }
TEST(ConSanDeviceEpochTest, RaceRetainedAcrossTagWrap) { run_epoch_boundary(1023, false, 2); }
TEST(ConSanDeviceEpochTest, AcrossTagWrapOrdered) { run_epoch_boundary(1023, true); }
TEST(ConSanDeviceEpochTest, AfterTagWrapOrdered) { run_epoch_boundary(1100, true); }
TEST(ConSanDeviceEpochTest, AfterTagWrapRace) { run_epoch_boundary(1100, false); }
TEST(ConSanDeviceEpochTest, OldEvidenceAtMatchingTagIsOrdered) {
  run_epoch_boundary(1023, true, 0, true);
}
TEST(ConSanDeviceEpochTest, Beyond16BitsOrdered) { run_epoch_boundary(65536, true); }
TEST(ConSanDeviceEpochTest, Beyond16BitsRace) { run_epoch_boundary(65536, false); }

} // namespace
