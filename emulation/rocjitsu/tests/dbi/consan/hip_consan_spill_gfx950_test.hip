// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_spill_gfx950_test.hip
/// @brief Native gfx950 validation of ConSan's CDNA4 VGPR spill contract.

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

__device__ __forceinline__ uint32_t scratch_roundtrip(uint32_t value) {
  asm volatile("scratch_store_dword off, %0, off offset:0\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               "v_mov_b32 %0, 0\n\t"
               "scratch_load_dword %0, off, off offset:0\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               : "+v"(value)
               :
               : "memory");
  return value;
}

__global__ __launch_bounds__(64) void gfx950_vgpr_scratch_roundtrip(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  volatile uint32_t private_words[4];
  private_words[tid & 3u] = tid;
  out[tid] = scratch_roundtrip(0x95000000u ^ tid);
}

__global__ __launch_bounds__(64) void gfx950_partial_exec_scratch_roundtrip(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  volatile uint32_t private_words[4];
  private_words[tid & 3u] = tid;
  uint32_t value = 0x95010000u ^ tid;
  if ((tid & 1u) == 0)
    value = scratch_roundtrip(value);
  out[tid] = value;
}

template <auto Kernel> void run_roundtrip(uint32_t expected_prefix) {
  constexpr uint32_t kThreads = 64;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kThreads * sizeof(uint32_t)));
  Kernel<<<1, kThreads>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(hipMemcpy(out.data(), device_out, kThreads * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], expected_prefix ^ tid) << "tid=" << tid;
}

TEST(ConSanSpillHipTest, Gfx950VgprScratchRoundTrip) {
  run_roundtrip<gfx950_vgpr_scratch_roundtrip>(0x95000000u);
}

TEST(ConSanSpillHipTest, Gfx950PartialExecVgprScratchRoundTrip) {
  run_roundtrip<gfx950_partial_exec_scratch_roundtrip>(0x95010000u);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
