// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Cross-wave publication through the native CDNA b64-to-b16 transpose read
/// used by Triton attention and matmul kernels. The exact 16-lane transpose
/// oracle makes both LDS addressing and the live destination state observable.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
using u32x2_t = uint32_t __attribute__((ext_vector_type(2)));

struct DeviceState {
  uint32_t synchronize;
  uint32_t result[kWaveSize][2];
  uint32_t control[kThreadCount];
};

__host__ __device__ constexpr uint32_t halfword_value(uint32_t lane, uint32_t halfword) {
  return (0x1200u + lane * 0x11u + halfword) & 0xffffu;
}

__host__ __device__ constexpr uint32_t pack_pair(uint32_t low, uint32_t high) {
  return (low & 0xffffu) | ((high & 0xffffu) << 16u);
}

__device__ __forceinline__ uint2 read_transposed(uint32_t address) {
  u32x2_t value;
  asm volatile("ds_read_b64_tr_b16 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_waitcnt lgkmcnt(0)\n\t"
               : "=&v"(value)
               : "v"(address)
               : "memory");
  return make_uint2(value[0], value[1]);
}

__global__
__launch_bounds__(kThreadCount) void cdna_transpose_publication_for_consan(DeviceState *state) {
  __shared__ __align__(8) uint2 tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);

  if (tid < kWaveSize) {
    tile[lane] = make_uint2(pack_pair(halfword_value(lane, 0u), halfword_value(lane, 1u)),
                            pack_pair(halfword_value(lane, 2u), halfword_value(lane, 3u)));
  }
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize) {
    const uint32_t address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&tile[lane]) & 0xffffu);
    const uint2 value = read_transposed(address);
    state->result[lane][0] = value.x;
    state->result[lane][1] = value.y;
  }
  state->control[tid] = 0x7a160000u | tid;
}

void run_cdna_transpose_publication(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  cdna_transpose_publication_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0x7a160000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;

  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    const uint32_t halfword = lane & 3u;
    const uint32_t source_base = (lane & 0x30u) + ((lane & 0x0cu) >> 2u);
    const uint32_t expected_low = pack_pair(halfword_value(source_base, halfword),
                                            halfword_value(source_base + 4u, halfword));
    const uint32_t expected_high = pack_pair(halfword_value(source_base + 8u, halfword),
                                             halfword_value(source_base + 12u, halfword));
    EXPECT_EQ(result.result[lane][0], expected_low) << "lane=" << lane;
    EXPECT_EQ(result.result[lane][1], expected_high) << "lane=" << lane;
  }
}

TEST(ConSanDeviceCdnaTransposePublicationTest, Correct) { run_cdna_transpose_publication(true); }

TEST(ConSanDeviceCdnaTransposePublicationTest, Incorrect) { run_cdna_transpose_publication(false); }

} // namespace
