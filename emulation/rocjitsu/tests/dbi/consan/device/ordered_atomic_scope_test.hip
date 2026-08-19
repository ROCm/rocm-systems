// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// LDS publication through a gfx12 instruction-scoped VGLOBAL atomic. The
/// broken member changes only the returning acquire atomic's encoded scope
/// from device to wave; address, operation, payload, cache operations, and
/// control flow remain identical.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumThreads = 64u;

struct DeviceState {
  uint32_t atomic_flag;
  uint32_t observed;
  uint32_t control[kMaximumThreads];
};

__device__ __forceinline__ void release_device_scope(uint32_t *flag) {
  const uint32_t one = 1u;
  asm volatile("s_wait_storecnt 0\n\t"
               "global_wb scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               "global_atomic_add_u32 %0, %1, off scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(flag), "v"(one)
               : "memory");
}

template <bool DeviceScope> __device__ __forceinline__ uint32_t acquire_scope(uint32_t *flag) {
  const uint32_t zero = 0u;
  uint32_t old = 0u;
  if constexpr (DeviceScope) {
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
                 "scope:SCOPE_CU\n\t"
                 "s_wait_storecnt 0\n\t"
                 "global_inv scope:SCOPE_DEV\n\t"
                 "s_wait_loadcnt 0\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
                 : "=&v"(old)
                 : "v"(flag), "v"(zero)
                 : "memory");
  }
  return old;
}

template <bool DeviceScope> __global__ void ordered_atomic_scope_for_consan(DeviceState *state) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;
  if (tid == 0u) {
    consan_device_test::store_lds_word(&payload, 0x5c0fe125u);
    release_device_scope(&state->atomic_flag);
  }
  if (tid == wave_size) {
    while (acquire_scope<DeviceScope>(&state->atomic_flag) == 0u) {
    }
    state->observed = consan_device_test::load_lds_word(&payload);
  }
  state->control[tid] = 0x5c000000u | tid;
}

void run_ordered_atomic_scope(bool device_scope) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreads);
  DeviceState initial{};
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  if (device_scope)
    ordered_atomic_scope_for_consan<true><<<1, thread_count>>>(device_state);
  else
    ordered_atomic_scope_for_consan<false><<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0x5c000000u | tid) << "thread=" << tid;
  EXPECT_EQ(result.atomic_flag, 1u);
  EXPECT_EQ(result.observed, 0x5c0fe125u);
}

TEST(ConSanDeviceOrderedAtomicScopeTest, Correct) { run_ordered_atomic_scope(true); }
TEST(ConSanDeviceOrderedAtomicScopeTest, Incorrect) { run_ordered_atomic_scope(false); }

} // namespace
