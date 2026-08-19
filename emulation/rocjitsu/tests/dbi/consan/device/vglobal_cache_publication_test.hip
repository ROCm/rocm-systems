// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// LDS publication through a target-native ordered VGLOBAL atomic and its
/// complete cache association. The broken member removes only the final
/// acquire-side cache operation; its atomic address, operation, payload,
/// scope, and control flow are otherwise identical.

#include "consan_device_test_support.h"

namespace {

#if defined(__gfx1250__)
#define RJ_VGLOBAL_ATOMIC_STORE_WAIT "s_wait_storecnt 0\n\t"
#else
#define RJ_VGLOBAL_ATOMIC_STORE_WAIT "s_waitcnt vmcnt(0)\n\t"
#endif

constexpr uint32_t kMaximumThreads = 128u;

struct DeviceState {
  uint32_t complete_cache_sequence;
  uint32_t atomic_flag;
  uint32_t observed;
  uint32_t control[kMaximumThreads];
};

__device__ __forceinline__ void release_vglobal(uint32_t *flag) {
  const uint32_t one = 1u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("buffer_wbl2 sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               "global_atomic_add %0, %1, off sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(flag), "v"(one)
               : "memory");
#elif defined(__gfx1100__)
  asm volatile("s_waitcnt_vscnt null, 0\n\t"
               "global_atomic_add_u32 %0, %1, off\n\t"
               "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(flag), "v"(one)
               : "memory");
#else
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "global_atomic_add_u32 %0, %1, off scope:SCOPE_DEV\n\t" RJ_VGLOBAL_ATOMIC_STORE_WAIT
                   RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(flag), "v"(one)
               : "memory");
#endif
}

template <bool CompleteCacheSequence>
__device__ __forceinline__ uint32_t acquire_vglobal(uint32_t *flag) {
  const uint32_t zero = 0u;
  uint32_t old = 0u;
#if defined(__gfx942__) || defined(__gfx950__)
  if constexpr (CompleteCacheSequence) {
    asm volatile("global_atomic_add %0, %1, %2, off sc0 sc1\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "buffer_inv sc1\n\t"
                 "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(flag), "v"(zero)
                 : "memory");
  } else {
    asm volatile("global_atomic_add %0, %1, %2, off sc0 sc1\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(flag), "v"(zero)
                 : "memory");
  }
#elif defined(__gfx1100__)
  if constexpr (CompleteCacheSequence) {
    asm volatile("global_atomic_add_u32 %0, %1, %2, off glc\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "buffer_gl1_inv\n\t"
                 "buffer_gl0_inv\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(flag), "v"(zero)
                 : "memory");
  } else {
    asm volatile("global_atomic_add_u32 %0, %1, %2, off glc\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 "buffer_gl1_inv\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(flag), "v"(zero)
                 : "memory");
  }
#else
  if constexpr (CompleteCacheSequence) {
    asm volatile("global_atomic_add_u32 %0, %1, %2, off th:TH_ATOMIC_RETURN "
                 "scope:SCOPE_DEV\n\t"
                 "s_wait_storecnt 0\n\t"
                 "global_inv scope:SCOPE_DEV\n\t"
                 "s_wait_loadcnt 0\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(flag), "v"(zero)
                 : "memory");
  } else {
    asm volatile("global_atomic_add_u32 %0, %1, %2, off th:TH_ATOMIC_RETURN "
                 "scope:SCOPE_DEV\n\t"
                 "s_wait_storecnt 0\n\t"
                 "s_wait_loadcnt 0\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(flag), "v"(zero)
                 : "memory");
  }
#endif
  return old;
}

template <bool CompleteCacheSequence>
__global__ void vglobal_cache_publication_for_consan(DeviceState *state) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&payload) & 0xffffu);
  if (tid == 0u) {
    consan_device_test::store_lds_word(&payload, 0xcace1250u);
    release_vglobal(&state->atomic_flag);
  }
  if (tid == wave_size) {
    while (acquire_vglobal<CompleteCacheSequence>(&state->atomic_flag) == 0u) {
    }
    state->observed = consan_device_test::load_lds_word(&payload);
  }
  state->control[tid] = 0xca000000u | tid;
  (void)address;
}

void run_vglobal_cache_publication(bool complete_cache_sequence) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  DeviceState initial{};
  initial.complete_cache_sequence = complete_cache_sequence ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  if (complete_cache_sequence)
    vglobal_cache_publication_for_consan<true><<<1, thread_count>>>(device_state);
  else
    vglobal_cache_publication_for_consan<false><<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0xca000000u | tid) << "thread=" << tid;
  if (complete_cache_sequence) {
    EXPECT_EQ(result.atomic_flag, 1u);
    EXPECT_EQ(result.observed, 0xcace1250u);
  }
}

TEST(ConSanDeviceVglobalCachePublicationTest, Correct) { run_vglobal_cache_publication(true); }
TEST(ConSanDeviceVglobalCachePublicationTest, Incorrect) { run_vglobal_cache_publication(false); }

} // namespace
