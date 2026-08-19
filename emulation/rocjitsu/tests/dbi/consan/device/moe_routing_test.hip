// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Top-1 MoE routing distilled from Aorta training and inference workloads.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kExpertCount = 4u;
constexpr uint32_t kMaximumTokens = 64u;
constexpr uint32_t kCanary = 0xdeadbeefu;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t token_count;
  uint32_t assignments[kMaximumTokens];
  uint32_t counts[kExpertCount];
  uint32_t offsets[kExpertCount + 1u];
  uint32_t scattered_tokens[kMaximumTokens + 2u];
  uint32_t gathered_outputs[kMaximumTokens];
  uint32_t control[128];
};

__device__ __host__ uint32_t route_token(uint32_t token) {
  // Expert two is deliberately empty. Uneven token counts still exercise all
  // other bins and their prefix offsets.
  const uint32_t route = token % 3u;
  return route == 2u ? 3u : route;
}

__device__ __host__ uint32_t token_value(uint32_t token) { return 7u + 3u * token; }

__device__ __host__ uint32_t expert_transform(uint32_t value, uint32_t expert) {
  // A small integer GLU-like gate/value product.
  return value * (value + expert + 1u);
}

__global__ __launch_bounds__(128) void moe_routing_for_consan(DeviceState *state) {
  __shared__ uint32_t bin_counts[kExpertCount];
  __shared__ uint32_t staged_routes[kMaximumTokens];
  __shared__ uint32_t staged_tokens[kMaximumTokens];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x7d000000u | tid;

  if (tid < kExpertCount)
    bin_counts[tid] = 0u;
  __syncthreads();

  if (wave == 1u && lane < state->token_count) {
    const uint32_t expert = route_token(lane);
    const uint32_t value = token_value(lane);
    staged_routes[lane] = expert;
    staged_tokens[lane] = value;
    state->assignments[lane] = expert;
    atomicAdd(&bin_counts[expert], 1u);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }

  // The incorrect member leaves routing decisions intact but removes the
  // publication of bin counts, routes, and staged tokens to the finalizer.
  if (state->synchronize)
    __syncthreads();

  if (tid == 0u) {
    uint32_t cursors[kExpertCount];
    uint32_t offset = 0u;
    for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
      const uint32_t count = bin_counts[expert];
      state->counts[expert] = count;
      state->offsets[expert] = offset;
      cursors[expert] = offset;
      offset += count;
    }
    state->offsets[kExpertCount] = offset;

    for (uint32_t token = 0; token < state->token_count; ++token) {
      const uint32_t expert = staged_routes[token];
      const uint32_t value = staged_tokens[token];
      const uint32_t position = cursors[expert]++;
      state->scattered_tokens[1u + position] = value;
      state->gathered_outputs[token] = expert_transform(value, expert);
    }
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

void run_moe_routing(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  const uint32_t token_count = wave_size - 5u;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  state.token_count = token_count;
  for (uint32_t &assignment : state.assignments)
    assignment = kCanary;
  for (uint32_t &value : state.scattered_tokens)
    value = kCanary;
  for (uint32_t &value : state.gathered_outputs)
    value = kCanary;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  moe_routing_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  EXPECT_EQ(state.token_count, token_count);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x7d000000u | tid) << "thread=" << tid;
  for (uint32_t token = 0; token < token_count; ++token)
    EXPECT_EQ(state.assignments[token], route_token(token)) << "token=" << token;
  for (uint32_t token = token_count; token < kMaximumTokens; ++token) {
    EXPECT_EQ(state.assignments[token], kCanary) << "assignment tail=" << token;
    EXPECT_EQ(state.gathered_outputs[token], kCanary) << "output tail=" << token;
  }
  EXPECT_EQ(state.scattered_tokens[0], kCanary);
  EXPECT_EQ(state.scattered_tokens[kMaximumTokens + 1u], kCanary);

  if (synchronize) {
    uint32_t expected_counts[kExpertCount] = {};
    uint32_t expected_offsets[kExpertCount + 1u] = {};
    for (uint32_t token = 0; token < token_count; ++token)
      ++expected_counts[route_token(token)];
    for (uint32_t expert = 0; expert < kExpertCount; ++expert)
      expected_offsets[expert + 1u] = expected_offsets[expert] + expected_counts[expert];

    uint32_t cursors[kExpertCount] = {};
    for (uint32_t expert = 0; expert < kExpertCount; ++expert)
      cursors[expert] = expected_offsets[expert];
    uint32_t expected_scatter[kMaximumTokens] = {};
    for (uint32_t token = 0; token < token_count; ++token) {
      const uint32_t expert = route_token(token);
      const uint32_t value = token_value(token);
      expected_scatter[cursors[expert]++] = value;
      EXPECT_EQ(state.gathered_outputs[token], expert_transform(value, expert))
          << "token=" << token;
    }

    for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
      EXPECT_EQ(state.counts[expert], expected_counts[expert]) << "expert=" << expert;
      EXPECT_EQ(state.offsets[expert], expected_offsets[expert]) << "expert=" << expert;
    }
    EXPECT_EQ(state.offsets[kExpertCount], token_count);
    EXPECT_EQ(state.counts[2], 0u);
    for (uint32_t position = 0; position < token_count; ++position)
      EXPECT_EQ(state.scattered_tokens[1u + position], expected_scatter[position])
          << "position=" << position;
    for (uint32_t position = token_count; position < kMaximumTokens; ++position)
      EXPECT_EQ(state.scattered_tokens[1u + position], kCanary) << "scatter tail=" << position;
  }
}

TEST(ConSanDeviceMoeRoutingTest, Correct) { run_moe_routing(true); }

TEST(ConSanDeviceMoeRoutingTest, Incorrect) { run_moe_routing(false); }

} // namespace
