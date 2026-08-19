// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Repeated ragged prefill/decode-shaped dispatches distilled from Aorta's
/// continuous-batching inference workload.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kTickCount = 4u;
constexpr uint32_t kRowCount = 7u;
constexpr uint32_t kMaximumThreads = 128u;
constexpr uint32_t kCanary = 0xcafef00du;
constexpr uint32_t kActiveRows[kTickCount] = {5u, 2u, 7u, 3u};

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t output[kTickCount][kRowCount];
  uint32_t control[kTickCount][kRowCount][kMaximumThreads];
};

__device__ __host__ uint32_t staged_value(uint32_t tick, uint32_t row, uint32_t lane) {
  return 100u * (tick + 1u) + 17u * row + lane + 1u;
}

__global__ __launch_bounds__(128) void continuous_batching_for_consan(DeviceState *state,
                                                                      uint32_t tick,
                                                                      uint32_t active_rows) {
  __shared__ uint32_t staged[64];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t row = blockIdx.x;
  state->control[tick][row][tid] = 0x7e000000u | (tick << 16u) | (row << 8u) | tid;

  // Masked inactive rows model scheduler slots with no request at this tick.
  if (row >= active_rows)
    return;

  // Every active row has a different partial-wave tail.
  const uint32_t valid = warpSize - 1u - ((3u * tick + 5u * row) % 9u);
  if (wave == 1u && lane < valid) {
    staged[lane] = staged_value(tick, row, lane);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }

  // The incorrect member preserves the launch schedule and masked rows but
  // removes the per-row publication edge.
  if (state->synchronize)
    __syncthreads();

  if (tid == 0u) {
    uint32_t sum = 0u;
    for (uint32_t producer = 0; producer < valid; ++producer)
      sum += staged[producer];
    state->output[tick][row] = sum;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

void run_sequence(bool synchronize, DeviceState *state) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  *state = {};
  state->synchronize = synchronize;
  state->wave_size = wave_size;
  for (auto &tick : state->output)
    for (uint32_t &value : tick)
      value = kCanary;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(*state)));
  HIP_ASSERT(hipMemcpy(device_state, state, sizeof(*state), hipMemcpyHostToDevice));
  for (uint32_t tick = 0; tick < kTickCount; ++tick) {
    continuous_batching_for_consan<<<kRowCount, thread_count>>>(device_state, tick,
                                                                kActiveRows[tick]);
    HIP_ASSERT(hipGetLastError());
  }
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(state, device_state, sizeof(*state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
}

uint64_t check_sequence(const DeviceState &state, bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  uint64_t checksum = 0u;
  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t tick = 0; tick < kTickCount; ++tick) {
    for (uint32_t row = 0; row < kRowCount; ++row) {
      for (uint32_t tid = 0; tid < thread_count; ++tid) {
        EXPECT_EQ(state.control[tick][row][tid], 0x7e000000u | (tick << 16u) | (row << 8u) | tid)
            << "tick=" << tick << " row=" << row << " thread=" << tid;
      }
      if (row >= kActiveRows[tick]) {
        EXPECT_EQ(state.output[tick][row], kCanary) << "tick=" << tick << " row=" << row;
        continue;
      }
      if (synchronize) {
        const uint32_t valid = wave_size - 1u - ((3u * tick + 5u * row) % 9u);
        uint32_t expected = 0u;
        for (uint32_t lane = 0; lane < valid; ++lane)
          expected += staged_value(tick, row, lane);
        EXPECT_EQ(state.output[tick][row], expected) << "tick=" << tick << " row=" << row;
        checksum = checksum * 1315423911u + state.output[tick][row];
      }
    }
  }
  return checksum;
}

void run_continuous_batching(bool synchronize) {
  // Execute the same scheduler trace twice from reset state. This verifies
  // stable repeated dispatch and prevents stale data or diagnostics from being
  // mistaken for a per-launch result.
  DeviceState first{};
  DeviceState second{};
  run_sequence(synchronize, &first);
  run_sequence(synchronize, &second);
  const uint64_t first_checksum = check_sequence(first, synchronize);
  const uint64_t second_checksum = check_sequence(second, synchronize);
  if (synchronize)
    EXPECT_EQ(first_checksum, second_checksum);
}

TEST(ConSanDeviceContinuousBatchingTest, Correct) { run_continuous_batching(true); }

TEST(ConSanDeviceContinuousBatchingTest, Incorrect) { run_continuous_batching(false); }

} // namespace
