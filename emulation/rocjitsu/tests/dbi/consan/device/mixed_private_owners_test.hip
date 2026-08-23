// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Two disconnected owners with different private-memory and register shapes.
///
/// This is the portable behavioral core of the CDNA full-bank mixed-state
/// fixture: a transformer must assign and restore state per kernel owner even
/// when one owner has a fixed private frame and the other has a runtime-sized
/// frame. The CDNA-only fixture retains its AccVGPR and exact physical-bank
/// stress; reproducing those details on targets that do not have the same
/// register file would not add meaningful coverage.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumWaveSize = 64u;
constexpr uint32_t kMaximumThreadCount = 2u * kMaximumWaveSize;

struct DeviceState {
  uint32_t synchronize_dynamic;
  uint32_t fixed_observed[kMaximumWaveSize];
  uint32_t dynamic_observed[kMaximumWaveSize];
  uint32_t fixed_private[kMaximumThreadCount];
  uint32_t dynamic_private[kMaximumThreadCount];
  uint32_t fixed_pressure[kMaximumThreadCount];
  uint32_t dynamic_pressure[kMaximumThreadCount];
  uint32_t fixed_control[kMaximumThreadCount];
  uint32_t dynamic_control[kMaximumThreadCount];
};

__device__ __forceinline__ uint32_t retain_fixed_vector_pressure(uint32_t seed) {
  uint32_t values[16];
#pragma unroll
  for (uint32_t index = 0u; index < 16u; ++index)
    values[index] =
        consan_device_test::rotate_left(seed ^ (0x9e3779b9u * (index + 1u)), (index & 7u) + 1u);
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]),
                 "+v"(values[8]), "+v"(values[9]), "+v"(values[10]), "+v"(values[11]),
                 "+v"(values[12]), "+v"(values[13]), "+v"(values[14]), "+v"(values[15]));
  uint32_t result = 0u;
#pragma unroll
  for (uint32_t index = 0u; index < 16u; ++index)
    result ^= values[index];
  return result;
}

__device__ __forceinline__ uint32_t retain_dynamic_vector_pressure(uint32_t seed) {
  uint32_t values[7];
#pragma unroll
  for (uint32_t index = 0u; index < 7u; ++index)
    values[index] = consan_device_test::rotate_left(seed + 0x1021u * index, index + 1u);
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]));
  uint32_t result = 0u;
#pragma unroll
  for (uint32_t index = 0u; index < 7u; ++index)
    result += values[index];
  return result;
}

__global__
__launch_bounds__(kMaximumThreadCount) void fixed_private_owner_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t tile[kMaximumWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  volatile uint32_t private_values[17];
#pragma unroll
  for (uint32_t index = 0u; index < 17u; ++index)
    private_values[index] = 0xf1000000u ^ (tid * 131u + index);
  const uint32_t selected_index = (tid * 5u + 3u) % 17u;
  const uint32_t selected = private_values[selected_index];
  const uint32_t pressure = retain_fixed_vector_pressure(0xf2000000u | tid);

  if (wave == 0u)
    consan_device_test::store_lds_word(&tile[lane], 0xf3000000u | lane);
  __syncthreads();
  if (wave == 1u)
    state->fixed_observed[lane] = consan_device_test::load_lds_word(&tile[lane]);

  state->fixed_private[tid] = selected;
  state->fixed_pressure[tid] = pressure;
  state->fixed_control[tid] = 0xf4000000u | tid;
}

__global__
__launch_bounds__(kMaximumThreadCount) void dynamic_private_owner_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t tile[2u * kMaximumWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t private_count = (tid & 15u) + 1u;
  auto *private_values =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  for (uint32_t index = 0u; index < private_count; ++index)
    private_values[index] = 0xd1000000u ^ (tid * 97u + index);
  const uint32_t selected_index = (tid * 7u + 5u) % private_count;
  const uint32_t selected = private_values[selected_index];
  const uint32_t pressure = retain_dynamic_vector_pressure(0xd2000000u | tid);

  if (wave == 0u)
    consan_device_test::store_lds_word(&tile[kMaximumWaveSize + lane], 0xd3000000u | lane);
  if (state->synchronize_dynamic != 0u)
    __syncthreads();
  if (wave == 1u)
    state->dynamic_observed[lane] =
        consan_device_test::load_lds_word(&tile[kMaximumWaveSize + lane]);

  state->dynamic_private[tid] = selected;
  state->dynamic_pressure[tid] = pressure;
  state->dynamic_control[tid] = 0xd4000000u | tid;
}

uint32_t fixed_private_reference(uint32_t tid) {
  return 0xf1000000u ^ (tid * 131u + (tid * 5u + 3u) % 17u);
}

uint32_t dynamic_private_reference(uint32_t tid) {
  const uint32_t count = (tid & 15u) + 1u;
  return 0xd1000000u ^ (tid * 97u + (tid * 7u + 5u) % count);
}

uint32_t fixed_pressure_reference(uint32_t tid) {
  const uint32_t seed = 0xf2000000u | tid;
  uint32_t result = 0u;
  for (uint32_t index = 0u; index < 16u; ++index)
    result ^=
        consan_device_test::rotate_left(seed ^ (0x9e3779b9u * (index + 1u)), (index & 7u) + 1u);
  return result;
}

uint32_t dynamic_pressure_reference(uint32_t tid) {
  const uint32_t seed = 0xd2000000u | tid;
  uint32_t result = 0u;
  for (uint32_t index = 0u; index < 7u; ++index)
    result += consan_device_test::rotate_left(seed + 0x1021u * index, index + 1u);
  return result;
}

void run_mixed_private_owners(bool synchronize_dynamic) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState initial{};
  initial.synchronize_dynamic = synchronize_dynamic ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  fixed_private_owner_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  dynamic_private_owner_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < thread_count; ++tid) {
    EXPECT_EQ(result.fixed_private[tid], fixed_private_reference(tid)) << "thread=" << tid;
    EXPECT_EQ(result.dynamic_private[tid], dynamic_private_reference(tid)) << "thread=" << tid;
    EXPECT_EQ(result.fixed_pressure[tid], fixed_pressure_reference(tid)) << "thread=" << tid;
    EXPECT_EQ(result.dynamic_pressure[tid], dynamic_pressure_reference(tid)) << "thread=" << tid;
    EXPECT_EQ(result.fixed_control[tid], 0xf4000000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.dynamic_control[tid], 0xd4000000u | tid) << "thread=" << tid;
  }
  for (uint32_t lane = 0u; lane < wave_size; ++lane) {
    EXPECT_EQ(result.fixed_observed[lane], 0xf3000000u | lane) << "lane=" << lane;
    if (synchronize_dynamic)
      EXPECT_EQ(result.dynamic_observed[lane], 0xd3000000u | lane) << "lane=" << lane;
  }
}

TEST(ConSanDeviceMixedPrivateOwnersTest, Correct) { run_mixed_private_owners(true); }

TEST(ConSanDeviceMixedPrivateOwnersTest, Incorrect) { run_mixed_private_owners(false); }

} // namespace
