// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// LDS publication through target-native FLAT atomic release/acquire forms.
/// This is distilled from the Hip-MOI Stream-K/tree workloads and the gfx1250
/// fault manifests. The incorrect member retains both atomic operations but
/// acquires an already-ready, unrelated object, so it imports no publication
/// edge for the producer's LDS payload.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumThreadCount = 128u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t release_counter;
  uint32_t unrelated_counter;
  uint32_t observed;
  uint32_t control[kMaximumThreadCount];
};

__device__ __forceinline__ void release_flat_atomic(uint32_t *counter) {
  const uint32_t one = 1u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("buffer_wbl2 sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               "flat_atomic_add %0, %1 sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(counter), "v"(one)
               : "memory");
#elif defined(__gfx1100__)
  asm volatile("s_waitcnt_vscnt null, 0\n\t"
               "flat_atomic_add_u32 %0, %1\n\t"
               "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(counter), "v"(one)
               : "memory");
#else
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "flat_atomic_add_u32 %0, %1 scope:SCOPE_DEV\n\t"
#if defined(__gfx1250__)
               "s_wait_storecnt 0\n\t"
#else
               "s_waitcnt vmcnt(0)\n\t"
#endif
               RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(counter), "v"(one)
               : "memory");
#endif
}

__device__ __forceinline__ uint32_t acquire_flat_atomic(uint32_t *counter) {
  const uint32_t zero = 0u;
  uint32_t old = 0u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("flat_atomic_add %0, %1, %2 sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               "buffer_inv sc1\n\t"
               "s_waitcnt vmcnt(0)\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               : "=&v"(old)
               : "v"(counter), "v"(zero)
               : "memory");
#elif defined(__gfx1100__)
  asm volatile("flat_atomic_add_u32 %0, %1, %2 glc\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               "buffer_gl1_inv\n\t"
               "buffer_gl0_inv\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               : "=&v"(old)
               : "v"(counter), "v"(zero)
               : "memory");
#else
  asm volatile("flat_atomic_add_u32 %0, %1, %2 th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "global_inv scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               : "=&v"(old)
               : "v"(counter), "v"(zero)
               : "memory");
#endif
  return old;
}

__global__
__launch_bounds__(kMaximumThreadCount) void flat_atomic_publication_for_consan(DeviceState *state) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / warpSize;
  const uint32_t lane = tid % warpSize;
  state->control[tid] = 0xfa000000u | tid;

  if (wave == 0u && lane == 0u) {
    consan_device_test::store_lds_word(&payload, 0xfa701250u);
    release_flat_atomic(&state->release_counter);
  }

  if (wave == 1u && lane == 0u) {
    uint32_t *counter =
        state->synchronize != 0u ? &state->release_counter : &state->unrelated_counter;
    while (acquire_flat_atomic(counter) == 0u) {
    }
    state->observed = consan_device_test::load_lds_word(&payload);
  }
}

void run_flat_atomic_publication(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreadCount);
  DeviceState state{};
  state.synchronize = synchronize ? 1u : 0u;
  state.unrelated_counter = 1u;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  flat_atomic_publication_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.release_counter, 1u);
  EXPECT_EQ(state.unrelated_counter, 1u);
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0xfa000000u | tid) << "thread=" << tid;
  if (synchronize)
    EXPECT_EQ(state.observed, 0xfa701250u);
}

TEST(ConSanDeviceFlatAtomicPublicationTest, Correct) { run_flat_atomic_publication(true); }

TEST(ConSanDeviceFlatAtomicPublicationTest, Incorrect) { run_flat_atomic_publication(false); }

} // namespace
