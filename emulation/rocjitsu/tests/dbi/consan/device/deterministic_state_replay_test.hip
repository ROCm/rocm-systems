// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// In-place state restore and deterministic replay distilled from Aorta's
/// activation-checkpoint and llm_determinism workloads. The same device
/// allocation is restored and reused for two mixed-precision
/// forward/backward/update generations.

#include "consan_device_test_support.h"

#include <bit>
#include <hip/hip_fp16.h>

namespace {

constexpr uint32_t kReplayCount = 2u;
constexpr uint32_t kMaximumLanes = 64u;
constexpr uint32_t kMaximumThreads = 128u;
constexpr uint32_t kCanaryBits = 0x4f123456u;
constexpr uint32_t kControlCanary = 0xdeadbeefu;
constexpr uint32_t kGuardBefore = 0x13579bdfu;
constexpr uint32_t kGuardAfter = 0x2468ace0u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t replay;
  uint32_t guard_before;
  float input[kMaximumLanes];
  float parameter[kMaximumLanes];
  float scratch[kMaximumLanes];
  float activation[kMaximumLanes];
  float gradient[kMaximumLanes];
  float updated_parameter[kMaximumLanes];
  uint32_t control[kMaximumThreads];
  uint32_t guard_after;
};

__host__ __device__ float input_value(uint32_t lane) {
  return static_cast<float>((lane & 7u) + 1u) * 0.125f;
}

__host__ __device__ float parameter_value(uint32_t lane) {
  return 4.0f + static_cast<float>(lane & 15u) * 0.125f;
}

__host__ __device__ float scratch_value(uint32_t lane) {
  return static_cast<float>((lane & 3u) + 1u) * 0.0625f;
}

__global__
__launch_bounds__(kMaximumThreads) void deterministic_state_replay_for_consan(DeviceState *state) {
  __shared__ __half checkpoint[kMaximumLanes];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0xac000000u | (state->replay << 16u) | tid;

  // Model a checkpointed activation written by one wave and recomputed by a
  // peer during backward. FP16 staging beside FP32 gradient/update state is
  // representative of the source workloads and keeps the exact oracle small.
  if (wave == 1u) {
    checkpoint[lane] =
        __float2half(state->input[lane] + 0.25f * state->parameter[lane] + state->scratch[lane]);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }

  // The adjacent incorrect member removes only this checkpoint-publication
  // edge. Host-side restore, allocation identity, math, and control traffic
  // are otherwise identical.
  if (state->synchronize)
    __syncthreads();

  if (wave == 0u) {
    const float activation = __half2float(checkpoint[lane]) + 0.5f * state->parameter[lane];
    const float gradient = 0.5f * activation + 0.25f * state->input[lane];
    const float updated = state->parameter[lane] - 0.125f * gradient;
    state->activation[lane] = activation;
    state->gradient[lane] = gradient;
    state->updated_parameter[lane] = updated;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

DeviceState make_initial_state(bool synchronize, uint32_t replay, uint32_t wave_size) {
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  state.replay = replay;
  state.guard_before = kGuardBefore;
  state.guard_after = kGuardAfter;
  const float canary = std::bit_cast<float>(kCanaryBits);
  for (uint32_t lane = 0u; lane < kMaximumLanes; ++lane) {
    state.input[lane] = input_value(lane);
    state.parameter[lane] = parameter_value(lane);
    state.scratch[lane] = scratch_value(lane);
    state.activation[lane] = canary;
    state.gradient[lane] = canary;
    state.updated_parameter[lane] = canary;
  }
  for (uint32_t &control : state.control)
    control = kControlCanary;
  return state;
}

void expect_float_bits(float actual, float expected, const char *field, uint32_t lane,
                       uint32_t replay) {
  EXPECT_EQ(std::bit_cast<uint32_t>(actual), std::bit_cast<uint32_t>(expected))
      << "field=" << field << " lane=" << lane << " replay=" << replay;
}

void check_replay(const DeviceState &state, bool synchronize, uint32_t replay, uint32_t wave_size) {
  const uint32_t thread_count = 2u * wave_size;
  EXPECT_EQ(state.synchronize, synchronize);
  EXPECT_EQ(state.wave_size, wave_size);
  EXPECT_EQ(state.replay, replay);
  EXPECT_EQ(state.guard_before, kGuardBefore);
  EXPECT_EQ(state.guard_after, kGuardAfter);

  for (uint32_t tid = 0u; tid < kMaximumThreads; ++tid) {
    if (tid < thread_count) {
      EXPECT_EQ(state.control[tid], 0xac000000u | (replay << 16u) | tid)
          << "thread=" << tid << " replay=" << replay;
    } else {
      EXPECT_EQ(state.control[tid], kControlCanary)
          << "inactive thread=" << tid << " replay=" << replay;
    }
  }

  const float canary = std::bit_cast<float>(kCanaryBits);
  for (uint32_t lane = 0u; lane < kMaximumLanes; ++lane) {
    const float input = input_value(lane);
    const float parameter = parameter_value(lane);
    const float scratch = scratch_value(lane);
    expect_float_bits(state.input[lane], input, "input", lane, replay);
    expect_float_bits(state.parameter[lane], parameter, "parameter", lane, replay);
    expect_float_bits(state.scratch[lane], scratch, "scratch", lane, replay);
    if (lane >= wave_size) {
      expect_float_bits(state.activation[lane], canary, "inactive activation", lane, replay);
      expect_float_bits(state.gradient[lane], canary, "inactive gradient", lane, replay);
      expect_float_bits(state.updated_parameter[lane], canary, "inactive parameter", lane, replay);
      continue;
    }
    if (synchronize) {
      const float checkpoint = __half2float(__float2half(input + 0.25f * parameter + scratch));
      const float activation = checkpoint + 0.5f * parameter;
      const float gradient = 0.5f * activation + 0.25f * input;
      const float updated = parameter - 0.125f * gradient;
      expect_float_bits(state.activation[lane], activation, "activation", lane, replay);
      expect_float_bits(state.gradient[lane], gradient, "gradient", lane, replay);
      expect_float_bits(state.updated_parameter[lane], updated, "updated parameter", lane, replay);
    }
  }
}

void run_deterministic_state_replay(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  ASSERT_LE(thread_count, kMaximumThreads);

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  DeviceState results[kReplayCount]{};
  for (uint32_t replay = 0u; replay < kReplayCount; ++replay) {
    const DeviceState initial = make_initial_state(synchronize, replay, wave_size);
    // Restore every input, parameter, scratch, output, control, and guard byte
    // into the same allocation before replaying the identical computation.
    HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(initial), hipMemcpyHostToDevice));
    deterministic_state_replay_for_consan<<<1, thread_count>>>(device_state);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());
    HIP_ASSERT(
        hipMemcpy(&results[replay], device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  }
  HIP_ASSERT(hipFree(device_state));

  for (uint32_t replay = 0u; replay < kReplayCount; ++replay)
    check_replay(results[replay], synchronize, replay, wave_size);

  if (synchronize) {
    for (uint32_t lane = 0u; lane < wave_size; ++lane) {
      expect_float_bits(results[1].activation[lane], results[0].activation[lane],
                        "replayed activation", lane, 1u);
      expect_float_bits(results[1].gradient[lane], results[0].gradient[lane], "replayed gradient",
                        lane, 1u);
      expect_float_bits(results[1].updated_parameter[lane], results[0].updated_parameter[lane],
                        "replayed parameter", lane, 1u);
    }
  }
}

TEST(ConSanDeviceDeterministicStateReplayTest, Correct) { run_deterministic_state_replay(true); }

TEST(ConSanDeviceDeterministicStateReplayTest, Incorrect) { run_deterministic_state_replay(false); }

} // namespace
