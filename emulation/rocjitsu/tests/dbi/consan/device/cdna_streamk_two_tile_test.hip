// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Five-phase, access-heavy LDS publication distilled from the gfx950
/// two-tile Stream-K workload. Each phase has eight distinct producer stores,
/// eight matching consumer loads, and one workgroup barrier. The incorrect
/// member removes only the middle publication edge while retaining the same
/// 80 static LDS access sites and five static barrier sites.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kPhaseCount = 5u;
constexpr uint32_t kSlotCount = 8u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t output[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_phase_word(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile("ds_write_b32 %0, %1\n\t" : : "v"(lds_address), "v"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_phase_word(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile("ds_read_b32 %0, %1\n\t"
               "s_waitcnt lgkmcnt(0)\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

__host__ __device__ __forceinline__ uint32_t phase_word(uint32_t phase, uint32_t slot,
                                                        uint32_t lane) {
  return 0xc7000000u ^ (phase << 16u) ^ (slot << 10u) ^ (lane * 17u + slot);
}

template <uint32_t Phase, uint32_t Slot>
__device__ __forceinline__ void publish_phase_slots(volatile uint32_t *storage, uint32_t lane) {
  store_phase_word(storage + Slot * kWaveSize + lane, phase_word(Phase, Slot, lane));
  if constexpr (Slot + 1u < kSlotCount)
    publish_phase_slots<Phase, Slot + 1u>(storage, lane);
}

template <uint32_t Phase, uint32_t Slot>
__device__ __forceinline__ void consume_phase_slots(volatile uint32_t *storage, uint32_t lane,
                                                    uint32_t &accumulator) {
  accumulator ^= consan_device_test::rotate_left(load_phase_word(storage + Slot * kWaveSize + lane),
                                                 (Phase + Slot) % 31u + 1u);
  if constexpr (Slot + 1u < kSlotCount)
    consume_phase_slots<Phase, Slot + 1u>(storage, lane, accumulator);
}

template <uint32_t Phase>
__device__ __forceinline__ void run_phase(volatile uint32_t *storage, DeviceState *state,
                                          uint32_t wave, uint32_t lane, uint32_t &accumulator) {
  volatile uint32_t *phase_storage = storage + Phase * kSlotCount * kWaveSize;
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
__launch_bounds__(kThreadCount) void cdna_streamk_two_tile_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t storage[kPhaseCount][kSlotCount][kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
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

void run_cdna_streamk_two_tile(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  cdna_streamk_two_tile_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0xc8000000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
      EXPECT_EQ(result.output[lane], lane == 0u ? expected_output(lane) : 0u) << "lane=" << lane;
  }
}

TEST(ConSanDeviceCdnaStreamkTwoTileTest, Correct) { run_cdna_streamk_two_tile(true); }

TEST(ConSanDeviceCdnaStreamkTwoTileTest, Incorrect) { run_cdna_streamk_two_tile(false); }

} // namespace
