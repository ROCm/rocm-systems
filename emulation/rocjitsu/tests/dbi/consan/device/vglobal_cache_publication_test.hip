// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// LDS publication through a target-native ordered VGLOBAL atomic and its
/// complete cache association. The broken member removes only the final
/// acquire-side cache operation; its atomic address, operation, payload,
/// scope, and control flow are otherwise identical. A second pair combines
/// that contract with an independent target-native workgroup barrier: split
/// signal/wait on RDNA4/CDNA5 and the equivalent monolithic barrier on earlier
/// targets. It is distilled from the gfx1250 Tensile TDM/Stream-K workloads.

#include "consan_device_test_support.h"

namespace {

#if defined(__gfx1250__)
#define RJ_VGLOBAL_ATOMIC_STORE_WAIT "s_wait_storecnt 0\n\t"
#else
#define RJ_VGLOBAL_ATOMIC_STORE_WAIT "s_waitcnt vmcnt(0)\n\t"
#endif

#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1100__)
#define RJ_VGLOBAL_WORKGROUP_BARRIER "s_barrier\n\t"
#else
#define RJ_VGLOBAL_WORKGROUP_BARRIER                                                               \
  "s_barrier_signal -1\n\t"                                                                        \
  "s_barrier_wait -1\n\t"
#endif

constexpr uint32_t kMaximumThreads = 128u;

struct DeviceState {
  uint32_t complete_cache_sequence;
  uint32_t atomic_flag;
  uint32_t observed;
  uint32_t control[kMaximumThreads];
};

constexpr uint32_t kCompositePayloadWords = 1u;

struct CompositeDeviceState {
  uint32_t atomic_flag;
  uint32_t barrier_observed;
  uint32_t payload_observed[kCompositePayloadWords];
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

__device__ __forceinline__ void target_workgroup_barrier() {
  asm volatile(RJ_VGLOBAL_WORKGROUP_BARRIER RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
}

#if !defined(RJ_CONSAN_DEVICE_FENCE_BARRIER_ONLY)

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

#else

template <bool CompleteCacheSequence>
__global__ void fence_barrier_publication_for_consan(CompositeDeviceState *state) {
  __shared__ uint32_t barrier_payload;
  __shared__ uint32_t published[kCompositePayloadWords];
  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;

  if (tid == 0u)
    consan_device_test::store_lds_word(&barrier_payload, 0xba220001u);
  target_workgroup_barrier();
  if (tid == wave_size)
    state->barrier_observed = consan_device_test::load_lds_word(&barrier_payload);

  if (tid == 0u) {
#pragma unroll
    for (uint32_t i = 0; i < kCompositePayloadWords; ++i)
      consan_device_test::store_lds_word(&published[i], 0xf1250000u | i);
    release_vglobal(&state->atomic_flag);
  }
  if (tid == wave_size) {
    while (acquire_vglobal<CompleteCacheSequence>(&state->atomic_flag) == 0u) {
    }
#pragma unroll
    for (uint32_t i = 0; i < kCompositePayloadWords; ++i)
      state->payload_observed[i] = consan_device_test::load_lds_word(&published[i]);
  }
  state->control[tid] = 0xfb000000u | tid;
}

#endif

#if !defined(RJ_CONSAN_DEVICE_FENCE_BARRIER_ONLY)

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

#else

void run_fence_barrier_publication(bool complete_cache_sequence) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  CompositeDeviceState initial{};
  CompositeDeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(CompositeDeviceState)));
  HIP_ASSERT(
      hipMemcpy(device_state, &initial, sizeof(CompositeDeviceState), hipMemcpyHostToDevice));
  if (complete_cache_sequence)
    fence_barrier_publication_for_consan<true><<<1, thread_count>>>(device_state);
  else
    fence_barrier_publication_for_consan<false><<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  CompositeDeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(CompositeDeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.barrier_observed, 0xba220001u);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0xfb000000u | tid) << "thread=" << tid;
  if (complete_cache_sequence) {
    EXPECT_EQ(result.atomic_flag, 1u);
    for (uint32_t i = 0; i < kCompositePayloadWords; ++i)
      EXPECT_EQ(result.payload_observed[i], 0xf1250000u | i) << "payload=" << i;
  }
}

#endif

#if !defined(RJ_CONSAN_DEVICE_FENCE_BARRIER_ONLY)

TEST(ConSanDeviceVglobalCachePublicationTest, Correct) { run_vglobal_cache_publication(true); }
TEST(ConSanDeviceVglobalCachePublicationTest, Incorrect) { run_vglobal_cache_publication(false); }

#else

TEST(ConSanDeviceFenceBarrierPublicationTest, Correct) { run_fence_barrier_publication(true); }
TEST(ConSanDeviceFenceBarrierPublicationTest, Incorrect) { run_fence_barrier_publication(false); }

#endif

} // namespace
