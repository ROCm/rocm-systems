// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Large-LDS, eight-wave pipeline distilled from the gfx950 HipKittens BF16
/// GEMM. It keeps the workload's 160 KiB group segment, 32 publication stages,
/// wide producer stores, scalar consumer reads, and repeated dispatches while
/// replacing MFMA arithmetic with a compact exact oracle.

#include "consan_device_test_support.h"

#include <array>

namespace {

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 8u * kWaveSize;
constexpr uint32_t kStages = 32u;
constexpr uint32_t kDispatches = 2u;
constexpr uint32_t kLdsBytes = 160u * 1024u;
constexpr uint32_t kStagedBytes = kStages * kWaveSize * sizeof(uint4);

struct LargeLdsStorage {
  uint4 staged[kStages][kWaveSize];
  uint8_t capacity_padding[kLdsBytes - kStagedBytes];
};

struct DeviceState {
  uint32_t output[kThreadCount];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_lds_b128(volatile uint4 *address, uint4 value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  register uint32_t value0 asm("v20") = value.x;
  register uint32_t value1 asm("v21") = value.y;
  register uint32_t value2 asm("v22") = value.z;
  register uint32_t value3 asm("v23") = value.w;
  asm volatile("ds_write_b128 %0, v[20:23]\n\t" RJ_CONSAN_DEVICE_CAVE
               :
               : "v"(lds_address), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "v20", "v21", "v22", "v23", "memory");
}

__device__ __forceinline__ uint32_t load_lds_b32(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile("ds_read_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_CAVE "s_waitcnt lgkmcnt(0)\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

__host__ __device__ __forceinline__ uint4 stage_value(uint32_t stage, uint32_t lane) {
  return make_uint4(0xb1000000u | (stage << 8u) | lane, 0xb2000000u | (stage << 8u) | lane,
                    0xb3000000u | (stage << 8u) | lane, 0xb4000000u | (stage << 8u) | lane);
}

template <bool Synchronize, uint32_t Stage>
__device__ __forceinline__ void run_large_lds_stage(volatile LargeLdsStorage *storage, uint32_t tid,
                                                    uint32_t lane, uint32_t &accumulator) {
  if (tid < kWaveSize)
    store_lds_b128(&storage->staged[Stage][lane], stage_value(Stage, lane));
  if constexpr (Synchronize || Stage != kStages / 2u)
    __syncthreads();

  volatile uint32_t *words =
      reinterpret_cast<volatile uint32_t *>(const_cast<uint4 *>(&storage->staged[Stage][lane]));
  const uint32_t first = load_lds_b32(words + 0u);
  const uint32_t second = load_lds_b32(words + 1u);
  const uint32_t third = load_lds_b32(words + 2u);
  accumulator ^= consan_device_test::rotate_left(first, (Stage % 13u) + 1u);
  accumulator ^= consan_device_test::rotate_left(second, (Stage % 17u) + 1u);
  accumulator ^= consan_device_test::rotate_left(third, (Stage % 19u) + 1u);

  if constexpr (Stage + 1u < kStages)
    run_large_lds_stage<Synchronize, Stage + 1u>(storage, tid, lane, accumulator);
}

template <bool Synchronize>
__global__
__launch_bounds__(kThreadCount) void gfx950_large_lds_pipeline_for_consan(DeviceState *state,
                                                                          uint32_t dispatch) {
  __shared__ LargeLdsStorage storage;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  if (tid == 0u)
    storage.capacity_padding[sizeof(storage.capacity_padding) - 1u] = 0x5au;
  __syncthreads();

  uint32_t accumulator = 0u;
  run_large_lds_stage<Synchronize, 0u>(&storage, tid, lane, accumulator);
  state->output[tid] = accumulator;
  state->control[tid] = 0xb5000000u | (dispatch << 12u) | tid;
}

uint32_t expected_output(uint32_t lane) {
  uint32_t accumulator = 0u;
  for (uint32_t stage = 0; stage < kStages; ++stage) {
    const uint4 value = stage_value(stage, lane);
    accumulator ^= consan_device_test::rotate_left(value.x, (stage % 13u) + 1u);
    accumulator ^= consan_device_test::rotate_left(value.y, (stage % 17u) + 1u);
    accumulator ^= consan_device_test::rotate_left(value.z, (stage % 19u) + 1u);
  }
  return accumulator;
}

void run_gfx950_large_lds_pipeline(bool synchronize) {
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemset(device_state, 0, sizeof(DeviceState)));
  for (uint32_t dispatch = 0; dispatch < kDispatches; ++dispatch) {
    if (synchronize) {
      gfx950_large_lds_pipeline_for_consan<true><<<1, kThreadCount>>>(device_state, dispatch);
    } else {
      gfx950_large_lds_pipeline_for_consan<false><<<1, kThreadCount>>>(device_state, dispatch);
    }
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());
  }

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(result), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
    EXPECT_EQ(result.control[tid], 0xb5000000u | ((kDispatches - 1u) << 12u) | tid)
        << "thread=" << tid;
    if (synchronize)
      EXPECT_EQ(result.output[tid], expected_output(tid & (kWaveSize - 1u))) << "thread=" << tid;
  }
}

TEST(ConSanDeviceGfx950LargeLdsPipelineTest, Correct) { run_gfx950_large_lds_pipeline(true); }

TEST(ConSanDeviceGfx950LargeLdsPipelineTest, Incorrect) { run_gfx950_large_lds_pipeline(false); }

} // namespace
