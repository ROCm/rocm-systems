// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace {

constexpr uint32_t kWaveSize = 64;
constexpr uint32_t kThreads = 4 * kWaveSize;

struct SimdResult {
  uint32_t low_bit, high_bit, simd_id, after_barrier, neighbor_value;
};

__global__ void read_simd_ids(SimdResult *output) {
  __shared__ uint32_t values[kThreads];
  uint32_t low_bit, high_bit, simd_id, after_barrier;
  // hipBLASLt selects its instruction schedule using the low SIMD_ID bit.
  asm volatile("s_getreg_b32 %0, hwreg(4, 4, 1)" : "=s"(low_bit));
  asm volatile("s_getreg_b32 %0, hwreg(4, 5, 1)" : "=s"(high_bit));
  asm volatile("s_getreg_b32 %0, hwreg(4, 4, 2)" : "=s"(simd_id));
  values[threadIdx.x] = 3 * threadIdx.x + 7;
  __syncthreads();
  asm volatile("s_getreg_b32 %0, hwreg(4, 4, 2)" : "=s"(after_barrier));
  const uint32_t neighbor = (threadIdx.x + kWaveSize) % kThreads;
  output[threadIdx.x] = {low_bit, high_bit, simd_id, after_barrier, values[neighbor]};
}

class HipHwIdTest : public ::testing::Test {
protected:
  SimdResult *device_results_ = nullptr;

  void TearDown() override {
    if (device_results_) {
      EXPECT_EQ(hipFree(device_results_), hipSuccess);
    }
  }
};

TEST_F(HipHwIdTest, FourWaveWorkgroupPreservesSimdIdsAcrossBarrier) {
  std::array<SimdResult, kThreads> results{};
  ASSERT_EQ(hipMalloc(&device_results_, sizeof(results)), hipSuccess);
  ASSERT_EQ(hipMemset(device_results_, 0xff, sizeof(results)), hipSuccess);

  // An idle CU assigns this workgroup's four waves to slots 0 through 3.
  read_simd_ids<<<1, kThreads>>>(device_results_);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(hipMemcpy(results.data(), device_results_, sizeof(results), hipMemcpyDeviceToHost),
            hipSuccess);

  for (uint32_t thread = 0; thread < kThreads; ++thread) {
    SCOPED_TRACE(thread);
    const uint32_t expected_simd = thread / kWaveSize;
    const auto &result = results[thread];
    EXPECT_EQ(result.low_bit, expected_simd & 1u);
    EXPECT_EQ(result.high_bit, expected_simd >> 1);
    EXPECT_EQ(result.simd_id, expected_simd);
    EXPECT_EQ(result.after_barrier, result.simd_id);
    // Each lane consumes LDS data written by a lane in the next wave.
    EXPECT_EQ(result.neighbor_value, 3 * ((thread + kWaveSize) % kThreads) + 7);
  }
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
