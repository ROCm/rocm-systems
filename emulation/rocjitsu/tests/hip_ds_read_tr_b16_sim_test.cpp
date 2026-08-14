// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_ds_read_tr_b16_sim_test.cpp
/// @brief Verify the ds_read_b64_tr_b16 cross-lane layout on gfx950.
///
/// One wave writes a known per-lane 4-halfword pattern to LDS with
/// ds_write_b64, reads it back with ds_read_b64_tr_b16, and checks every lane
/// against the 4x16-lane halfword transpose: destination lane l halfword n
/// comes from source lane ((l & 0x30) | ((l & 0xc) >> 2)) + 4 * n, halfword
/// (l & 3).

#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

constexpr int kWaveSize = 64;

using U32x2 = uint32_t __attribute__((ext_vector_type(2)));

constexpr uint32_t tr_b16_halfword_value(int thread, int halfword) {
  return (0x1200u + static_cast<uint32_t>(thread) * 0x11u + static_cast<uint32_t>(halfword)) &
         0xffffu;
}

constexpr uint32_t pack_u16_pair(uint32_t lo, uint32_t hi) {
  return (lo & 0xffffu) | ((hi & 0xffffu) << 16);
}

__global__ __launch_bounds__(kWaveSize) void ds_read_tr_b16_kernel(uint32_t *out) {
  __shared__ uint32_t lds[kWaveSize * 2];
  const int tid = threadIdx.x;
  const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) + tid * 8;
  U32x2 raw = {pack_u16_pair(tr_b16_halfword_value(tid, 0), tr_b16_halfword_value(tid, 1)),
               pack_u16_pair(tr_b16_halfword_value(tid, 2), tr_b16_halfword_value(tid, 3))};
  U32x2 value = {};
  asm volatile("ds_write_b64 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(raw)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b64_tr_b16 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[tid * 2 + 0] = value[0];
  out[tid * 2 + 1] = value[1];
}

} // namespace

TEST(DsReadTrB16SimTest, LaneLayout) {
  constexpr int kOutWords = kWaveSize * 2;
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, kOutWords * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, kOutWords * sizeof(uint32_t)));

  ds_read_tr_b16_kernel<<<1, kWaveSize>>>(device_out);
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> host_out(kOutWords);
  HIP_ASSERT(
      hipMemcpy(host_out.data(), device_out, kOutWords * sizeof(uint32_t), hipMemcpyDeviceToHost));
  (void)hipFree(device_out);

  for (int lane = 0; lane < kWaveSize; ++lane) {
    const int halfword = lane & 3;
    const int source_base = (lane & 0x30) + ((lane & 0x0c) >> 2);
    const uint32_t expected_lo = pack_u16_pair(tr_b16_halfword_value(source_base + 0, halfword),
                                               tr_b16_halfword_value(source_base + 4, halfword));
    const uint32_t expected_hi = pack_u16_pair(tr_b16_halfword_value(source_base + 8, halfword),
                                               tr_b16_halfword_value(source_base + 12, halfword));
    EXPECT_EQ(host_out[lane * 2 + 0], expected_lo) << "lane " << lane << " low dword";
    EXPECT_EQ(host_out[lane * 2 + 1], expected_hi) << "lane " << lane << " high dword";
  }
}
