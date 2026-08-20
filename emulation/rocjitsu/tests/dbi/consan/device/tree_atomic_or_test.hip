// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Tree Stream-K-style bitmask publication through release/acquire atomic OR.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_count;
  uint32_t completion[2];
  uint32_t final_mask;
  uint32_t observed;
  uint32_t control[256];
};

__global__ __launch_bounds__(256) void tree_atomic_or_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t partials[4];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x77000000u | tid;

  if (lane != 0)
    return;

  consan_device_test::store_lds_word(&partials[wave], 0x200u + 19u * wave);

  // Waves 1, 2, and 3 deliberately race to publish distinct release edges
  // through one direct-mapped atomic object. This is the small checked-in
  // form of the concurrent producer pattern in Hip-MOI's Stream-K tree E2E
  // workload. The final wave must not turn ordinary publication contention
  // into ConSan overflow or lose any producer's causal edge.
  if (wave != 0u) {
    (void)__hip_atomic_fetch_or(&state->completion[0], 1u << wave, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_AGENT);
  } else {
    const uint32_t producer_mask = ((1u << state->wave_count) - 1u) & ~1u;
    while ((__hip_atomic_load(&state->completion[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) &
            producer_mask) != producer_mask) {
    }
    uint32_t *completion = &state->completion[state->synchronize ? 0u : 1u];
    state->final_mask =
        __hip_atomic_fetch_or(completion, 0u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    uint32_t sum = 0u;
#pragma clang loop unroll(disable)
    for (uint32_t rank = 0; rank < state->wave_count; ++rank) {
      // Visit a different wave's cell first. Sampled tracks one dynamic cell
      // at this static load site, so leading with the reader's own cell would
      // hide the intentionally unordered pair in the incorrect workload.
      const uint32_t producer = (rank + 1u) % state->wave_count;
      sum += consan_device_test::load_lds_word(&partials[producer]);
    }
    state->observed = sum;
  }
}

void run_tree_atomic_or(bool synchronize) {
  const uint32_t thread_count = 4u * consan_device_test::active_wave_size();
  const uint32_t wave_count = thread_count / consan_device_test::active_wave_size();
  ASSERT_EQ(wave_count, 4u);
  ASSERT_LE(thread_count, 256u);
  const uint32_t producer_mask = ((1u << wave_count) - 1u) & ~1u;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_count = wave_count;
  state.completion[1] = producer_mask;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  tree_atomic_or_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(state.wave_count, wave_count);
  EXPECT_EQ(state.completion[0], producer_mask);
  EXPECT_EQ(state.completion[1], producer_mask);
  EXPECT_EQ(state.final_mask, producer_mask);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x77000000u | tid) << "thread=" << tid;
  if (synchronize)
    EXPECT_EQ(state.observed, wave_count * 0x200u + 19u * wave_count * (wave_count - 1u) / 2u);
}

TEST(ConSanDeviceTreeAtomicOrTest, Correct) { run_tree_atomic_or(true); }

TEST(ConSanDeviceTreeAtomicOrTest, Incorrect) { run_tree_atomic_or(false); }

} // namespace
