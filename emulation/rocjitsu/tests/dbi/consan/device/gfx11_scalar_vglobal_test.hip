// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// gfx11 scalar-base VGLOBAL acquire coverage. The vector-only address form is
/// covered by vglobal_cache_publication_test; this adjacent pair holds the
/// global pointer in an SGPR pair and supplies a zero VGPR offset. The broken
/// member removes only the final GL0 acquire invalidation.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumThreads = 64u;

struct DeviceState {
  uint32_t atomic_flag;
  uint32_t observed;
  uint32_t control[kMaximumThreads];
};

__device__ __forceinline__ void release_vector_vglobal(uint32_t *flag) {
  const uint32_t one = 1u;
  asm volatile("s_waitcnt_vscnt null, 0\n\t"
               "global_atomic_add_u32 %0, %1, off\n\t"
               "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(flag), "v"(one)
               : "memory");
}

template <bool CompleteCacheSequence>
__device__ __forceinline__ uint32_t acquire_scalar_vglobal(uint32_t *flag) {
  const uint32_t zero = 0u;
  const uint32_t zero_offset = 0u;
  uint32_t old = 0u;
  if constexpr (CompleteCacheSequence) {
    asm volatile("global_atomic_add_u32 %0, %1, %2, %3 glc\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "buffer_gl1_inv\n\t"
                 "buffer_gl0_inv\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(zero_offset), "v"(zero), "s"(flag)
                 : "memory");
  } else {
    asm volatile("global_atomic_add_u32 %0, %1, %2, %3 glc\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "buffer_gl1_inv\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(zero_offset), "v"(zero), "s"(flag)
                 : "memory");
  }
  return old;
}

template <bool CompleteCacheSequence>
__global__ void gfx11_scalar_vglobal_for_consan(DeviceState *state) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;
  if (tid == 0u) {
    consan_device_test::store_lds_word(&payload, 0x5ca1ab1eu);
    release_vector_vglobal(&state->atomic_flag);
  }
  if (tid == wave_size) {
    while (acquire_scalar_vglobal<CompleteCacheSequence>(&state->atomic_flag) == 0u) {
    }
    state->observed = consan_device_test::load_lds_word(&payload);
  }
  state->control[tid] = 0x5b000000u | tid;
}

void run_gfx11_scalar_vglobal(bool complete_cache_sequence) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreads);
  DeviceState initial{};
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  if (complete_cache_sequence)
    gfx11_scalar_vglobal_for_consan<true><<<1, thread_count>>>(device_state);
  else
    gfx11_scalar_vglobal_for_consan<false><<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0x5b000000u | tid) << "thread=" << tid;
  EXPECT_EQ(result.atomic_flag, 1u);
  EXPECT_EQ(result.observed, 0x5ca1ab1eu);
}

TEST(ConSanDeviceGfx11ScalarVglobalTest, Correct) { run_gfx11_scalar_vglobal(true); }
TEST(ConSanDeviceGfx11ScalarVglobalTest, Incorrect) { run_gfx11_scalar_vglobal(false); }

} // namespace
