// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Atomic-store state publication distilled from framework and RCCL-style
/// producer/consumer protocols.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t phase[2];
  uint32_t cas_succeeded;
  uint32_t cas_expected;
  uint32_t observed;
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void atomic_store_publication_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t payload[8];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x6e000000u | tid;

  if (wave == 0u && lane == 0u) {
#pragma clang loop unroll(disable)
    for (uint32_t i = 0u; i < 8u; ++i)
      consan_device_test::store_lds_word(&payload[i], 0x280u + 19u * i);

    const uint32_t phase_index = state->synchronize ? 0u : 1u;
    __hip_atomic_store(&state->phase[phase_index], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }

  if (wave == 1u && lane == 0u) {
    uint32_t expected = 1u;
    do {
      expected = 1u;
      state->cas_succeeded =
          __hip_atomic_compare_exchange_strong(&state->phase[0], &expected, 1u, __ATOMIC_ACQUIRE,
                                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (!state->cas_succeeded);
    state->cas_expected = expected;

    uint32_t sum = 0u;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0u; i < 8u; ++i)
      sum += consan_device_test::load_lds_word(&payload[i]);
    state->observed = sum;
  }
}

void run_atomic_store_publication(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, 128u);

  DeviceState state{};
  state.synchronize = synchronize;
  // The incorrect member publishes a distinct phase object while the
  // consumer's phase is already ready. It retains the same release-store and
  // acquire-CAS shape without importing the producer's LDS publication edge.
  state.phase[0] = synchronize ? 0u : 1u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  atomic_store_publication_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.phase[0], 1u);
  EXPECT_EQ(state.phase[1], synchronize ? 0u : 1u);
  EXPECT_EQ(state.cas_succeeded, 1u);
  EXPECT_EQ(state.cas_expected, 1u);
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x6e000000u | tid) << "thread=" << tid;
  if (synchronize)
    EXPECT_EQ(state.observed, 8u * 0x280u + 19u * 28u);
}

TEST(ConSanDeviceAtomicStorePublicationTest, Correct) { run_atomic_store_publication(true); }

TEST(ConSanDeviceAtomicStorePublicationTest, Incorrect) { run_atomic_store_publication(false); }

} // namespace
