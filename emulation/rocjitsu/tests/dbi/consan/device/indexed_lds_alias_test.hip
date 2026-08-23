// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Runtime-computed LDS chunk addressing distilled from indexed gather/scatter
/// and split-tile reductions. The adjacent members use the same compiled store
/// site and synchronization; only the runtime address stride changes from one
/// to zero, collapsing every logical slot onto one physical address.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kProducerWaveCount = 2u;
constexpr uint32_t kTotalWaveCount = 3u;
constexpr uint32_t kMaximumLanes = 64u;
constexpr uint32_t kMaximumSlots = kProducerWaveCount * kMaximumLanes;
constexpr uint32_t kMaximumThreads = kTotalWaveCount * kMaximumLanes;
constexpr uint32_t kObservedCanary = 0xa5a55a5au;
constexpr uint32_t kControlCanary = 0xdeadbeefu;

struct DeviceState {
  uint32_t address_stride;
  uint32_t wave_size;
  uint32_t observed[kMaximumSlots];
  uint32_t control[kMaximumThreads];
};

__host__ __device__ uint32_t producer_value(uint32_t producer, uint32_t lane) {
  return 0x31000000u | (producer << 20u) | lane;
}

__host__ __device__ uint32_t producer_lane_for_slot(uint32_t producer, uint32_t slot) {
  // The second producer's pair exchange retains the runtime-index flavor of
  // selection and scatter kernels while remaining exactly invertible.
  return producer == 0u ? slot : slot ^ 1u;
}

__global__
__launch_bounds__(kMaximumThreads) void indexed_lds_alias_for_consan(DeviceState *state) {
  __shared__ uint32_t chunks[kMaximumSlots];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x76000000u | tid;

  // Give the intentionally unpopulated logical slots an exact oracle in the
  // incorrect member. Each thread initializes distinct slots.
  for (uint32_t slot = tid; slot < 2u * warpSize; slot += blockDim.x)
    chunks[slot] = kObservedCanary;
  __syncthreads();

  if (wave < kProducerWaveCount) {
    const uint32_t chunk_slot = producer_lane_for_slot(wave, lane);
    const uint32_t logical_slot = wave * warpSize + chunk_slot;
    chunks[state->address_stride * logical_slot] = producer_value(wave, lane);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }

  // Both members retain this publication edge. The incorrect member is racy
  // because both producer waves wrote the first chunk in the same epoch, not
  // because a barrier was removed.
  __syncthreads();

  if (wave == 2u) {
    state->observed[lane] = chunks[lane];
    state->observed[warpSize + lane] = chunks[warpSize + lane];
  }
}

void run_indexed_lds_alias(bool collapse_addresses) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = kTotalWaveCount * wave_size;
  ASSERT_LE(thread_count, kMaximumThreads);

  DeviceState state{};
  state.address_stride = collapse_addresses ? 0u : 1u;
  state.wave_size = wave_size;
  for (uint32_t &value : state.observed)
    value = kObservedCanary;
  for (uint32_t &control : state.control)
    control = kControlCanary;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  indexed_lds_alias_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.address_stride, collapse_addresses ? 0u : 1u);
  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t tid = 0u; tid < kMaximumThreads; ++tid) {
    if (tid < thread_count)
      EXPECT_EQ(state.control[tid], 0x76000000u | tid) << "thread=" << tid;
    else
      EXPECT_EQ(state.control[tid], kControlCanary) << "inactive thread=" << tid;
  }

  for (uint32_t slot = 0u; slot < kMaximumSlots; ++slot) {
    if (slot >= 2u * wave_size) {
      EXPECT_EQ(state.observed[slot], kObservedCanary) << "inactive slot=" << slot;
      continue;
    }
    if (!collapse_addresses) {
      const uint32_t producer = slot >= wave_size ? 1u : 0u;
      const uint32_t chunk_slot = slot - producer * wave_size;
      EXPECT_EQ(state.observed[slot],
                producer_value(producer, producer_lane_for_slot(producer, chunk_slot)))
          << "slot=" << slot;
    } else if (slot != 0u) {
      EXPECT_EQ(state.observed[slot], kObservedCanary) << "untouched slot=" << slot;
    }
  }
}

TEST(ConSanDeviceIndexedLdsAliasTest, Correct) { run_indexed_lds_alias(false); }

TEST(ConSanDeviceIndexedLdsAliasTest, Incorrect) { run_indexed_lds_alias(true); }

} // namespace
