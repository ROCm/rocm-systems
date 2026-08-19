// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Atomic publication handoff distilled from Stream-K.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t handoff[2];
  uint32_t instrumentation_ready;
  uint32_t release_ticket;
  uint32_t acquire_ticket;
  uint32_t observed;
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void atomic_arrival_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t partials[8];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x75000000u + tid;

  if (wave == 0 && lane == 0) {
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 8u; ++i)
      consan_device_test::store_lds_word(&partials[i], 0xc0ffee00u + i);
    state->release_ticket =
        __hip_atomic_fetch_add(&state->handoff[0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    // Release instrumentation publishes after the displaced atomic. Keep the
    // consumer out of its acquire probe until that trampoline has returned;
    // this readiness word is not itself the LDS synchronization contract.
    *reinterpret_cast<volatile uint32_t *>(&state->instrumentation_ready) = 1u;
  }
  if (wave == 1 && lane == 0) {
    volatile uint32_t *ready = &state->instrumentation_ready;
    while (*ready == 0u) {
    }
    if (state->synchronize) {
      state->acquire_ticket = __hip_atomic_fetch_add(&state->handoff[0], 1u, __ATOMIC_ACQ_REL,
                                                     __HIP_MEMORY_SCOPE_AGENT);
    } else {
      // Acquiring a different atomic does not import the producer's LDS
      // writes. This is the small-scale wrong-address failure seen in
      // split-tile and Stream-K publication protocols.
      state->acquire_ticket = __hip_atomic_fetch_add(&state->handoff[1], 1u, __ATOMIC_ACQ_REL,
                                                     __HIP_MEMORY_SCOPE_AGENT);
    }
    uint32_t sum = 0u;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 8u; ++i)
      sum += consan_device_test::load_lds_word(&partials[i]);
    state->observed = sum;
  }
}

void run_atomic_arrival(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  atomic_arrival_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.handoff[0], synchronize ? 2u : 1u);
  EXPECT_EQ(state.handoff[1], synchronize ? 0u : 1u);
  EXPECT_EQ(state.instrumentation_ready, 1u);
  EXPECT_EQ(state.release_ticket, 0u);
  EXPECT_EQ(state.acquire_ticket, synchronize ? 1u : 0u);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x75000000u + tid) << "thread=" << tid;
  if (synchronize)
    EXPECT_EQ(state.observed, 8u * 0xc0ffee00u + 28u);
}

TEST(ConSanDeviceAtomicArrivalTest, Correct) { run_atomic_arrival(true); }

TEST(ConSanDeviceAtomicArrivalTest, Incorrect) { run_atomic_arrival(false); }

} // namespace
