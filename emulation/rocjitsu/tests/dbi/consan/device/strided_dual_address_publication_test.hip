// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Stride-64 dual-address B64 LDS publication distilled from Triton attention
/// and Sharktank kernels. The two accesses are 4 KiB apart, matching the
/// offset1:8 form in the retained E2E assembly corpus.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumWaveSize = 64u;
constexpr uint32_t kMaximumThreadCount = 2u * kMaximumWaveSize;
constexpr uint32_t kStride64Offset = 8u;
constexpr uint32_t kSecondRangeByteOffset = kStride64Offset * 64u * sizeof(uint64_t);
constexpr uint32_t kLdsByteCount = kSecondRangeByteOffset + kMaximumWaveSize * sizeof(uint64_t);

struct DeviceState {
  uint32_t synchronize;
  uint64_t first[kMaximumWaveSize];
  uint64_t second[kMaximumWaveSize];
  uint32_t control[kMaximumThreadCount];
};

struct LdsPair {
  uint64_t first;
  uint64_t second;
};

__host__ __device__ uint64_t expected_value(uint32_t range, uint32_t lane) {
  return (range == 0u ? 0x7100000000000000ull : 0x7200000000000000ull) |
         (static_cast<uint64_t>(lane) << 32u) | (0x13570000u + 17u * lane + range);
}

__device__ __forceinline__ void store_strided_pair(uint32_t address, uint64_t first,
                                                   uint64_t second) {
  register uint32_t value0 asm("v20") = static_cast<uint32_t>(first);
  register uint32_t value1 asm("v21") = static_cast<uint32_t>(first >> 32u);
  register uint32_t value2 asm("v22") = static_cast<uint32_t>(second);
  register uint32_t value3 asm("v23") = static_cast<uint32_t>(second >> 32u);
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("ds_write2st64_b64 %[address], v[20:21], v[22:23] offset0:0 "
               "offset1:8\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : [address] "v"(address), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "v20", "v21", "v22", "v23", "memory");
#else
  asm volatile("ds_store_2addr_stride64_b64 %[address], v[20:21], v[22:23] offset0:0 "
               "offset1:8\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : [address] "v"(address), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "v20", "v21", "v22", "v23", "memory");
#endif
}

__device__ __forceinline__ LdsPair load_strided_pair(uint32_t address) {
  register uint32_t value0 asm("v20") = 0u;
  register uint32_t value1 asm("v21") = 0u;
  register uint32_t value2 asm("v22") = 0u;
  register uint32_t value3 asm("v23") = 0u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile(
      "ds_read2st64_b64 v[20:23], %[address] offset0:0 offset1:8\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
      "s_waitcnt lgkmcnt(0)\n\t"
      : "+v"(value0), "+v"(value1), "+v"(value2), "+v"(value3)
      : [address] "v"(address)
      : "memory");
#elif defined(__gfx1201__) || defined(__gfx1250__)
  asm volatile("ds_load_2addr_stride64_b64 v[20:23], %[address] offset0:0 "
               "offset1:8\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               : "+v"(value0), "+v"(value1), "+v"(value2), "+v"(value3)
               : [address] "v"(address)
               : "memory");
#else
  asm volatile("ds_load_2addr_stride64_b64 v[20:23], %[address] offset0:0 "
               "offset1:8\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_waitcnt lgkmcnt(0)\n\t"
               : "+v"(value0), "+v"(value1), "+v"(value2), "+v"(value3)
               : [address] "v"(address)
               : "memory");
#endif
  return {static_cast<uint64_t>(value0) | (static_cast<uint64_t>(value1) << 32u),
          static_cast<uint64_t>(value2) | (static_cast<uint64_t>(value3) << 32u)};
}

__global__ void strided_dual_address_publication_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint8_t storage[kLdsByteCount];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t address = static_cast<uint32_t>(
      reinterpret_cast<uintptr_t>(&storage[lane * sizeof(uint64_t)]) & 0xffffu);

  state->control[tid] = 0x73000000u | tid;
  if (wave == 0u)
    store_strided_pair(address, expected_value(0u, lane), expected_value(1u, lane));
  if (state->synchronize)
    __syncthreads();
  if (wave == 1u) {
    const LdsPair pair = load_strided_pair(address);
    state->first[lane] = pair.first;
    state->second[lane] = pair.second;
  }
}

void run_strided_dual_address_publication(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  strided_dual_address_publication_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x73000000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0u; lane < wave_size; ++lane) {
      EXPECT_EQ(state.first[lane], expected_value(0u, lane)) << "range=0 lane=" << lane;
      EXPECT_EQ(state.second[lane], expected_value(1u, lane)) << "range=1 lane=" << lane;
    }
  }
}

TEST(ConSanDeviceStridedDualAddressPublicationTest, Correct) {
  run_strided_dual_address_publication(true);
}

TEST(ConSanDeviceStridedDualAddressPublicationTest, Incorrect) {
  run_strided_dual_address_publication(false);
}

} // namespace
