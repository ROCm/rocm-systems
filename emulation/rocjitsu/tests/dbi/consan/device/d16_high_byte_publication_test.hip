// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// D16-high byte publication distilled from sparse-FP8 matrix staging. The
/// producer stores adjacent low and high source halves, and the peer wave
/// consumes the high byte through a D16-high destination. Target-native
/// spellings differ, but the exact byte-range contract applies everywhere.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumWaveSize = 64u;
constexpr uint32_t kMaximumThreadCount = 2u * kMaximumWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t packed[kMaximumWaveSize];
  uint32_t control[kMaximumThreadCount];
};

__device__ __forceinline__ void store_d16_high_bytes(uint32_t address) {
  register uint32_t packed asm("v20") = 0x5a5a00a5u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("ds_write_b8 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "ds_write_b8_d16_hi %0, %1 offset:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(address), "v"(packed)
               : "memory");
#else
  asm volatile("ds_store_b8 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "ds_store_b8_d16_hi %0, %1 offset:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(address), "v"(packed)
               : "memory");
#endif
}

__device__ __forceinline__ uint32_t load_d16_high_bytes(uint32_t address) {
  register uint32_t packed asm("v20") = 0u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("ds_read_u8_d16_hi %0, %1 offset:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_waitcnt lgkmcnt(0)\n\t"
               : "+v"(packed)
               : "v"(address)
               : "memory");
#elif defined(__gfx1100__)
  asm volatile("ds_load_u8_d16_hi %0, %1 offset:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_waitcnt lgkmcnt(0)\n\t"
               : "+v"(packed)
               : "v"(address)
               : "memory");
#else
  asm volatile("ds_load_u8_d16_hi %0, %1 offset:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_wait_dscnt 0\n\t"
               : "+v"(packed)
               : "v"(address)
               : "memory");
#endif
  // D16-high destination operations preserve the low half on RDNA but clear
  // it on CDNA. Normalize that architecturally permitted representation and
  // validate the published high byte identically on every target.
  return (packed >> 16u) & 0xffu;
}

__global__ __launch_bounds__(kMaximumThreadCount) void d16_high_byte_publication_for_consan(
    DeviceState *state) {
  __shared__ __align__(4) uint8_t bytes[2u * kMaximumWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&bytes[2u * lane]) & 0xffffu);

  if (wave == 0u)
    store_d16_high_bytes(address);
  if (state->synchronize != 0u)
    __syncthreads();
  if (wave == 1u)
    state->packed[lane] = load_d16_high_bytes(address);
  state->control[tid] = 0xd1600000u | tid;
}

void run_d16_high_byte_publication(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  initial.wave_size = wave_size;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  d16_high_byte_publication_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.wave_size, wave_size);
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0xd1600000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0u; lane < wave_size; ++lane)
      EXPECT_EQ(result.packed[lane], 0x0000005au) << "lane=" << lane;
  }
}

TEST(ConSanDeviceD16HighBytePublicationTest, Correct) { run_d16_high_byte_publication(true); }

TEST(ConSanDeviceD16HighBytePublicationTest, Incorrect) { run_d16_high_byte_publication(false); }

} // namespace
