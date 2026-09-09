// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Same-kernel, same-kernarg dispatches for two explicit HSA queues. The host
/// fixture submits packet zero on both queues so their AMDHSA dispatch-ID
/// preloads collide by construction.

#include <hip/hip_runtime.h>

#include <cstdint>

#define RJ_CONSAN_GFX950_TWO_QUEUE_CAVE                                                         \
  ".rept 512\n\t"                                                                              \
  "s_nop 0\n\t"                                                                                \
  ".endr\n\t"

namespace {

constexpr uint32_t kMaximumThreads = 128u;
constexpr uint32_t kDispatches = 4u;

struct TwoQueueState {
  uint32_t launches;
  uint32_t observed[kDispatches];
  uint32_t control[kDispatches][kMaximumThreads];
};

__device__ __forceinline__ void store_lds_word(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile("ds_write_b32 %0, %1\n\t" RJ_CONSAN_GFX950_TWO_QUEUE_CAVE
               :
               : "v"(lds_address), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_lds_word(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile("ds_read_b32 %0, %1\n\t" RJ_CONSAN_GFX950_TWO_QUEUE_CAVE
               "s_waitcnt lgkmcnt(0)\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

template <bool Synchronize> __device__ void run_two_queue_dispatch(TwoQueueState *state) {
  __shared__ uint32_t payload;
  __shared__ uint32_t launch_slot;
  __shared__ uint32_t owner_seed;
  const uint32_t tid = threadIdx.x;

  if (tid == 0u)
    launch_slot = atomicAdd(&state->launches, 1u);
  __syncthreads();

  const uint32_t launch = launch_slot;
  if constexpr (Synchronize) {
    // Each launch is race-free by itself, but they deliberately use the same
    // LDS byte from different wave owners. If ConSan aliases dispatch zero on
    // the two queues, replay manufactures a cross-dispatch conflict even
    // though separate dispatches cannot share an LDS allocation.
    // Force the first wave to acquire ConSan owner zero in both dispatches;
    // otherwise a lone active second wave can itself be numbered owner zero.
    if (tid == 0u)
      store_lds_word(&owner_seed, 0xe9520000u | launch);
    if ((launch & 1u) == 0u && tid == 0u) {
      store_lds_word(&payload, 0xe9500000u | launch);
      state->observed[launch] = 0xe9500000u | launch;
    }
    if ((launch & 1u) != 0u && tid == warpSize) {
      store_lds_word(&payload, 0xe9500000u | launch);
      state->observed[launch] = load_lds_word(&payload);
    }
  } else {
    if (tid == 0u)
      store_lds_word(&payload, 0xe9500000u | launch);
    if (launch < kDispatches && tid == warpSize)
      state->observed[launch] = load_lds_word(&payload);
  }
  if (launch < kDispatches)
    state->control[launch][tid] = 0xe9510000u | (launch << 8u) | tid;
}

} // namespace

extern "C" __global__ __launch_bounds__(128) void
consan_gfx950_two_queue_dispatch_correct(TwoQueueState *state) {
  run_two_queue_dispatch<true>(state);
}

extern "C" __global__ __launch_bounds__(128) void
consan_gfx950_two_queue_dispatch_incorrect(TwoQueueState *state) {
  run_two_queue_dispatch<false>(state);
}
