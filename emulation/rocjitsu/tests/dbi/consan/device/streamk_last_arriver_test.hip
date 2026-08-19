// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Stream-K-style partial publication through a fetch-add last-arriver chain.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_count;
  uint32_t turn;
  uint32_t handoff[2];
  uint32_t tickets[4];
  uint32_t observed;
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void streamk_last_arriver_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t partials[4];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x76000000u | tid;

  if (lane != 0)
    return;

  volatile uint32_t *turn = &state->turn;
  while (*turn != wave) {
  }
  consan_device_test::store_lds_word(&partials[wave], 0x100u + 17u * wave);

  uint32_t *handoff = &state->handoff[0];
  if (!state->synchronize && wave + 1u == state->wave_count)
    handoff = &state->handoff[1];
  const uint32_t ticket =
      __hip_atomic_fetch_add(handoff, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  state->tickets[wave] = ticket;
  *turn = wave + 1u;

  // The fault fixture primes the wrong counter so that the same last-arriver
  // branch executes without acquiring the other waves' publications.
  if (ticket + 1u == state->wave_count) {
    uint32_t sum = 0u;
#pragma clang loop unroll(disable)
    for (uint32_t producer = 0; producer < state->wave_count; ++producer)
      sum += consan_device_test::load_lds_word(&partials[producer]);
    state->observed = sum;
  }
}

void run_streamk_last_arriver(bool synchronize) {
  const uint32_t thread_count = 128u;
  const uint32_t wave_count = thread_count / consan_device_test::active_wave_size();
  ASSERT_TRUE(wave_count == 2u || wave_count == 4u);
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_count = wave_count;
  state.handoff[1] = wave_count - 1u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  streamk_last_arriver_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(state.wave_count, wave_count);
  EXPECT_EQ(state.turn, wave_count);
  EXPECT_EQ(state.handoff[0], synchronize ? wave_count : wave_count - 1u);
  EXPECT_EQ(state.handoff[1], synchronize ? wave_count - 1u : wave_count);
  for (uint32_t wave = 0; wave < wave_count; ++wave)
    EXPECT_EQ(state.tickets[wave], wave) << "wave=" << wave;
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x76000000u | tid) << "thread=" << tid;
  if (synchronize)
    EXPECT_EQ(state.observed, wave_count * 0x100u + 17u * wave_count * (wave_count - 1u) / 2u);
}

TEST(ConSanDeviceStreamKLastArriverTest, Correct) { run_streamk_last_arriver(true); }

TEST(ConSanDeviceStreamKLastArriverTest, Incorrect) { run_streamk_last_arriver(false); }

} // namespace
