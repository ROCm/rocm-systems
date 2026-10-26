// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    const hipError_t error = (call);                                                               \
    ASSERT_EQ(error, hipSuccess) << "HIP error: " << hipGetErrorString(error);                     \
  } while (0)

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

struct LocationResult {
  uint32_t wave_simd, cu_shader_engine, xcc, smid;
};

__global__ void read_location_ids(LocationResult *output) {
  uint32_t wave_simd, cu_shader_engine, xcc;
  asm volatile("s_getreg_b32 %0, hwreg(4, 0, 6)" : "=s"(wave_simd));
  asm volatile("s_getreg_b32 %0, hwreg(4, 8, 8)" : "=s"(cu_shader_engine));
  asm volatile("s_getreg_b32 %0, hwreg(20)" : "=s"(xcc));
  output[blockIdx.x * blockDim.x + threadIdx.x] = {wave_simd, cu_shader_engine, xcc, __smid()};
}

class HipHwIdTest : public ::testing::Test {
protected:
  SimdResult *device_results_ = nullptr;
  LocationResult *device_locations_ = nullptr;

  void TearDown() override {
    if (device_locations_) {
      const hipError_t error = hipFree(device_locations_);
      EXPECT_EQ(error, hipSuccess) << "HIP error: " << hipGetErrorString(error);
    }
    if (device_results_) {
      const hipError_t error = hipFree(device_results_);
      EXPECT_EQ(error, hipSuccess) << "HIP error: " << hipGetErrorString(error);
    }
  }
};

TEST_F(HipHwIdTest, FourWaveWorkgroupPreservesSimdIdsAcrossBarrier) {
  std::array<SimdResult, kThreads> results{};
  HIP_ASSERT(hipMalloc(&device_results_, sizeof(results)));
  HIP_ASSERT(hipMemset(device_results_, 0xff, sizeof(results)));

  // An idle CU assigns this workgroup's four waves to slots 0 through 3.
  read_simd_ids<<<1, kThreads>>>(device_results_);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(results.data(), device_results_, sizeof(results), hipMemcpyDeviceToHost));

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

TEST_F(HipHwIdTest, MultiXcdDispatchReportsConsistentLocation) {
  // CTest supplies the bounds from the selected simulator configuration.
  const char *xcd_env = std::getenv("RJ_HW_ID_XCDS");
  const char *se_env = std::getenv("RJ_HW_ID_SHADER_ENGINES");
  const char *cu_env = std::getenv("RJ_HW_ID_CUS_PER_SE");
  ASSERT_NE(xcd_env, nullptr);
  ASSERT_NE(se_env, nullptr);
  ASSERT_NE(cu_env, nullptr);
  const uint32_t xcd_count = std::atoi(xcd_env);
  const uint32_t se_count = std::atoi(se_env);
  const uint32_t cu_count = std::atoi(cu_env);
  ASSERT_GT(xcd_count, 1u);
  ASSERT_LE(xcd_count, 16u);
  ASSERT_GT(se_count, 0u);
  ASSERT_LE(se_count, 8u);
  ASSERT_GT(cu_count, 0u);
  ASSERT_LE(cu_count, 16u);

  constexpr uint32_t kBlocks = 64;
  std::vector<LocationResult> results(kBlocks * kThreads);
  const size_t bytes = results.size() * sizeof(LocationResult);
  HIP_ASSERT(hipMalloc(&device_locations_, bytes));
  HIP_ASSERT(hipMemset(device_locations_, 0xff, bytes));
  read_location_ids<<<kBlocks, kThreads>>>(device_locations_);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(results.data(), device_locations_, bytes, hipMemcpyDeviceToHost));

  std::set<uint32_t> seen_xcds;
  for (size_t thread = 0; thread < results.size(); ++thread) {
    SCOPED_TRACE(thread);
    const auto &result = results[thread];
    const uint32_t wave = result.wave_simd & 15u;
    const uint32_t simd = result.wave_simd >> 4;
    const uint32_t cu = result.cu_shader_engine & 15u;
    const uint32_t shader_array = (result.cu_shader_engine >> 4) & 1u;
    const uint32_t se = result.cu_shader_engine >> 5;
    EXPECT_LT(wave, 8u); // These configurations have 32 resident slots across four SIMDs.
    EXPECT_LT(simd, 4u);
    EXPECT_LT(cu, cu_count);
    EXPECT_EQ(shader_array, 0u);
    EXPECT_LT(se, se_count);
    EXPECT_LT(result.xcc, xcd_count);
    // HIP packs two SE bits and four CU bits below the XCC index.
    EXPECT_EQ(result.smid, (result.xcc << 6) | (se << 4) | cu);
    const auto &leader = results[thread - thread % kWaveSize];
    EXPECT_EQ(result.wave_simd, leader.wave_simd);
    EXPECT_EQ(result.cu_shader_engine, leader.cu_shader_engine);
    EXPECT_EQ(result.xcc, leader.xcc);
    seen_xcds.insert(result.xcc);
  }
  // Catch an all-zero XCC implementation without imposing a workgroup order.
  EXPECT_GT(seen_xcds.size(), 1u);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
