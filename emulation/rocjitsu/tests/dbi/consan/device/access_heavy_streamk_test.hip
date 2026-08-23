// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Five-phase, access-heavy LDS publication distilled from the gfx950
/// two-tile Stream-K workload. Each phase has eight distinct producer stores,
/// eight matching consumer loads, and one workgroup barrier. The incorrect
/// member removes only the middle publication edge while retaining the same
/// 80 static LDS access sites and five static barrier sites. The contract is
/// wave-size independent and runs on every supported architecture.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumWaveSize = 64u;
constexpr uint32_t kMaximumThreadCount = 2u * kMaximumWaveSize;
constexpr uint32_t kPhaseCount = 5u;
constexpr uint32_t kSlotCount = 8u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t output[kMaximumWaveSize];
  uint32_t control[kMaximumThreadCount];
};

__host__ __device__ __forceinline__ uint32_t phase_word(uint32_t phase, uint32_t slot,
                                                        uint32_t lane) {
  return 0xc7000000u ^ (phase << 16u) ^ (slot << 10u) ^ (lane * 17u + slot);
}

template <uint32_t Phase, uint32_t Slot>
__device__ __forceinline__ void publish_phase_slots(volatile uint32_t *storage, uint32_t lane) {
  consan_device_test::store_lds_word_without_cave(storage + Slot * kMaximumWaveSize + lane,
                                                  phase_word(Phase, Slot, lane));
  if constexpr (Slot + 1u < kSlotCount)
    publish_phase_slots<Phase, Slot + 1u>(storage, lane);
}

template <uint32_t Phase, uint32_t Slot>
__device__ __forceinline__ void consume_phase_slots(volatile uint32_t *storage, uint32_t lane,
                                                    uint32_t &accumulator) {
  accumulator ^= consan_device_test::rotate_left(
      consan_device_test::load_lds_word_without_cave(storage + Slot * kMaximumWaveSize + lane),
      (Phase + Slot) % 31u + 1u);
  if constexpr (Slot + 1u < kSlotCount)
    consume_phase_slots<Phase, Slot + 1u>(storage, lane, accumulator);
}

template <uint32_t Phase>
__device__ __forceinline__ void run_phase(volatile uint32_t *storage, DeviceState *state,
                                          uint32_t wave, uint32_t lane, uint32_t &accumulator) {
  volatile uint32_t *phase_storage = storage + Phase * kSlotCount * kMaximumWaveSize;
  if (wave == 0u && lane == 0u)
    publish_phase_slots<Phase, 0u>(phase_storage, lane);

  // Uniformly skip only the central publication edge in the incorrect run.
  // The barrier remains a static site in the shared code object.
  if (Phase != kPhaseCount / 2u || state->synchronize != 0u)
    __syncthreads();

  if (wave == 1u && lane == 0u)
    consume_phase_slots<Phase, 0u>(phase_storage, lane, accumulator);
  if constexpr (Phase + 1u < kPhaseCount)
    run_phase<Phase + 1u>(storage, state, wave, lane, accumulator);
}

__global__
__launch_bounds__(kMaximumThreadCount) void access_heavy_streamk_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t storage[kPhaseCount][kSlotCount][kMaximumWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0xc8000000u | tid;

  uint32_t accumulator = 0u;
  run_phase<0u>(&storage[0][0][0], state, wave, lane, accumulator);
  if (wave == 1u)
    state->output[lane] = accumulator;
}

uint32_t expected_output(uint32_t lane) {
  uint32_t accumulator = 0u;
  for (uint32_t phase = 0u; phase < kPhaseCount; ++phase) {
    for (uint32_t slot = 0u; slot < kSlotCount; ++slot) {
      accumulator ^=
          consan_device_test::rotate_left(phase_word(phase, slot, lane), (phase + slot) % 31u + 1u);
    }
  }
  return accumulator;
}

void run_access_heavy_streamk(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  initial.wave_size = wave_size;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  access_heavy_streamk_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.wave_size, wave_size);
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0xc8000000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0u; lane < wave_size; ++lane)
      EXPECT_EQ(result.output[lane], lane == 0u ? expected_output(lane) : 0u) << "lane=" << lane;
  }
}

TEST(ConSanDeviceAccessHeavyStreamKTest, Correct) { run_access_heavy_streamk(true); }

TEST(ConSanDeviceAccessHeavyStreamKTest, Incorrect) { run_access_heavy_streamk(false); }

} // namespace
