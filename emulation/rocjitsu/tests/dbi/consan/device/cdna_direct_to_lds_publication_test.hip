// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Direct global-to-LDS publication distilled from the gfx950 HipKittens BF16
/// GEMM. One wave issues native MUBUF loads directly into LDS; a peer wave
/// consumes the staged values. Adjacent members differ only in the workgroup
/// publication barrier, while the correct member checks every transferred
/// value and both members retain an independent control oracle.

#include "consan_device_test_support.h"

#include <array>

namespace {

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t observed[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void
direct_global_to_lds_b32(const uint32_t *source, uint32_t byte_offset, uint32_t wave_byte_offset) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(source);
  const uint32_t descriptor0 = static_cast<uint32_t>(base);
  const uint32_t descriptor1 = static_cast<uint32_t>(base >> 32u) & 0xffffu;
  const uint32_t descriptor2 = kWaveSize * sizeof(uint32_t);
  const uint32_t descriptor3 = 4u | (7u << 12u) | (4u << 15u);
  asm volatile("s_mov_b32 s12, %0\n\t"
               "s_mov_b32 s13, %1\n\t"
               "s_mov_b32 s14, %2\n\t"
               "s_mov_b32 s15, %3\n\t"
               "s_mov_b32 m0, %5\n\t"
               "buffer_load_dword %4, s[12:15], %6 offen lds\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_waitcnt vmcnt(0)\n\t"
               :
               : "s"(descriptor0), "s"(descriptor1), "s"(descriptor2), "s"(descriptor3),
                 "v"(byte_offset), "s"(wave_byte_offset), "s"(wave_byte_offset)
               : "s12", "s13", "s14", "s15", "memory");
}

template <bool Synchronize>
__global__ __launch_bounds__(kThreadCount) void cdna_direct_to_lds_publication_for_consan(
    const uint32_t *input, DeviceState *state) {
  __shared__ uint32_t staged[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave_byte_offset =
      __builtin_amdgcn_readfirstlane((tid / kWaveSize) * kWaveSize * sizeof(uint32_t));
  if (tid < kWaveSize)
    direct_global_to_lds_b32(input, lane * sizeof(uint32_t), wave_byte_offset);

  if constexpr (Synchronize)
    __syncthreads();
  else
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");

  if (tid >= kWaveSize)
    state->observed[lane] = consan_device_test::load_lds_word(&staged[lane]);
  state->control[tid] = 0xd7100000u | tid;
}

void run_cdna_direct_to_lds_publication(bool synchronize) {
  std::array<uint32_t, kWaveSize> input{};
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    input[lane] = 0xd7200000u | (lane * 17u + 3u);

  uint32_t *device_input = nullptr;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_input, sizeof(input)));
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_input, input.data(), sizeof(input), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(device_state, 0, sizeof(DeviceState)));
  if (synchronize) {
    cdna_direct_to_lds_publication_for_consan<true>
        <<<1, kThreadCount>>>(device_input, device_state);
  } else {
    cdna_direct_to_lds_publication_for_consan<false>
        <<<1, kThreadCount>>>(device_input, device_state);
  }
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(result), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  HIP_ASSERT(hipFree(device_input));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0xd7100000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      EXPECT_EQ(result.observed[lane], input[lane]) << "lane=" << lane;
  }
}

TEST(ConSanDeviceCdnaDirectToLdsPublicationTest, Correct) {
  run_cdna_direct_to_lds_publication(true);
}

TEST(ConSanDeviceCdnaDirectToLdsPublicationTest, Incorrect) {
  run_cdna_direct_to_lds_publication(false);
}

} // namespace
