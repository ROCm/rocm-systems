// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Mixed-precision gradient publication and optimizer update distilled from
/// transformer training workloads.

#include "consan_device_test_support.h"

#include <bit>
#include <hip/hip_fp16.h>

namespace {

constexpr uint32_t kProducerWaveCount = 3u;
constexpr uint32_t kTotalWaveCount = kProducerWaveCount + 1u;
constexpr uint32_t kMaximumLanes = 64u;
constexpr uint32_t kMaximumThreads = kTotalWaveCount * kMaximumLanes;
constexpr uint32_t kCanaryBits = 0x4f234567u;
constexpr uint32_t kControlCanary = 0xdeadbeefu;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  float reduced_gradient[kMaximumLanes];
  float first_moment[kMaximumLanes];
  float updated_parameter[kMaximumLanes];
  uint32_t control[kMaximumThreads];
};

__host__ __device__ float local_gradient_value(uint32_t lane) {
  return static_cast<float>((lane & 7u) + 1u) * 0.03125f;
}

__host__ __device__ float partial_gradient_value(uint32_t producer, uint32_t lane) {
  return static_cast<float>(8u * (producer + 1u) + (lane & 7u) + 1u) * 0.0625f;
}

__host__ __device__ uint32_t producer_valid_lanes(uint32_t wave_size, uint32_t producer) {
  return wave_size - (3u + 2u * producer);
}

__global__
__launch_bounds__(kMaximumThreads) void backward_optimizer_for_consan(DeviceState *state) {
  // Three producer waves publish ragged FP16 partial gradients. A fourth wave
  // performs the hierarchical FP32 reduction and optimizer update. This keeps
  // the mixed-precision and partial-tail shape of generated training kernels
  // without requiring a model asset.
  __shared__ __half published_gradient[kProducerWaveCount][kMaximumLanes];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x7c000000u | tid;

  if (wave >= 1u) {
    const uint32_t producer = wave - 1u;
    if (lane < producer_valid_lanes(warpSize, producer)) {
      published_gradient[producer][lane] = __float2half(partial_gradient_value(producer, lane));
      asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
    }
  }

  // The incorrect member removes only this collective publication edge. It
  // retains all three producer waves, their tails, the FP16 traffic, the FP32
  // reduction, and the optimizer math.
  if (state->synchronize)
    __syncthreads();

  if (wave == 0u) {
    float reduced = local_gradient_value(lane);
#pragma unroll
    for (uint32_t producer = 0u; producer < kProducerWaveCount; ++producer) {
      if (lane < producer_valid_lanes(warpSize, producer))
        reduced += __half2float(published_gradient[producer][lane]);
    }
    const float old_moment = static_cast<float>(lane & 3u) * 0.25f;
    const float moment = 0.5f * old_moment + 0.5f * reduced;
    const float parameter = 4.0f + static_cast<float>(lane & 7u) * 0.125f;
    // A compact AdamW-like update: a first-moment term and decoupled weight
    // decay. All operands are binary fractions so the host oracle is exact.
    const float updated = parameter - 0.25f * moment - 0.125f * parameter;
    state->reduced_gradient[lane] = reduced;
    state->first_moment[lane] = moment;
    state->updated_parameter[lane] = __half2float(__float2half(updated));
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

DeviceState make_initial_state(bool synchronize, uint32_t wave_size) {
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  const float canary = std::bit_cast<float>(kCanaryBits);
  for (uint32_t lane = 0u; lane < kMaximumLanes; ++lane) {
    state.reduced_gradient[lane] = canary;
    state.first_moment[lane] = canary;
    state.updated_parameter[lane] = canary;
  }
  for (uint32_t &control : state.control)
    control = kControlCanary;
  return state;
}

void expect_float_bits(float actual, float expected, const char *field, uint32_t lane) {
  EXPECT_EQ(std::bit_cast<uint32_t>(actual), std::bit_cast<uint32_t>(expected))
      << "field=" << field << " lane=" << lane;
}

void run_backward_optimizer(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = kTotalWaveCount * wave_size;
  ASSERT_LE(thread_count, kMaximumThreads);
  DeviceState state = make_initial_state(synchronize, wave_size);
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  backward_optimizer_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.synchronize, synchronize);
  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t tid = 0u; tid < kMaximumThreads; ++tid) {
    if (tid < thread_count)
      EXPECT_EQ(state.control[tid], 0x7c000000u | tid) << "thread=" << tid;
    else
      EXPECT_EQ(state.control[tid], kControlCanary) << "inactive thread=" << tid;
  }

  const float canary = std::bit_cast<float>(kCanaryBits);
  for (uint32_t lane = 0u; lane < kMaximumLanes; ++lane) {
    if (lane >= wave_size) {
      expect_float_bits(state.reduced_gradient[lane], canary, "inactive gradient", lane);
      expect_float_bits(state.first_moment[lane], canary, "inactive moment", lane);
      expect_float_bits(state.updated_parameter[lane], canary, "inactive parameter", lane);
      continue;
    }
    if (synchronize) {
      float reduced = local_gradient_value(lane);
      for (uint32_t producer = 0u; producer < kProducerWaveCount; ++producer) {
        if (lane < producer_valid_lanes(wave_size, producer))
          reduced += __half2float(__float2half(partial_gradient_value(producer, lane)));
      }
      const float old_moment = static_cast<float>(lane & 3u) * 0.25f;
      const float moment = 0.5f * old_moment + 0.5f * reduced;
      const float parameter = 4.0f + static_cast<float>(lane & 7u) * 0.125f;
      const float updated = parameter - 0.25f * moment - 0.125f * parameter;
      expect_float_bits(state.reduced_gradient[lane], reduced, "reduced gradient", lane);
      expect_float_bits(state.first_moment[lane], moment, "first moment", lane);
      expect_float_bits(state.updated_parameter[lane], __half2float(__float2half(updated)),
                        "updated parameter", lane);
    }
  }
}

TEST(ConSanDeviceBackwardOptimizerTest, Correct) { run_backward_optimizer(true); }

TEST(ConSanDeviceBackwardOptimizerTest, Incorrect) { run_backward_optimizer(false); }

} // namespace
