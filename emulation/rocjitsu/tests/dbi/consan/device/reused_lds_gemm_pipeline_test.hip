// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Same-storage LDS staging distilled from Qwen's generated output matmul.
/// The kernel publishes one tile, consumes it, waits for every reader before
/// overwriting the same LDS addresses, and publishes the replacement tile.
/// The adjacent members differ only in the reader-retirement barrier: the
/// correct member checks both stages and the fused result exactly, while the
/// incorrect member must be diagnosed by the instrumented CTest contract.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumLanes = 64u;
constexpr uint32_t kMaximumThreads = 128u;

struct DeviceState {
  uint32_t synchronize_reuse;
  uint32_t wave_size;
  uint32_t stage_zero[kMaximumLanes];
  uint32_t stage_one[kMaximumLanes];
  uint32_t fused[kMaximumLanes];
  uint32_t control[kMaximumThreads];
};

__host__ __device__ __forceinline__ uint32_t stage_zero_value(uint32_t lane) {
  return 0x51000000u ^ (lane * 0x1021u);
}

__host__ __device__ __forceinline__ uint32_t stage_one_value(uint32_t lane) {
  return 0xa7000000u ^ (lane * 0x45d9u);
}

__host__ __device__ __forceinline__ uint32_t fuse_stages(uint32_t first, uint32_t second,
                                                         uint32_t lane) {
  return consan_device_test::rotate_left(first ^ second ^ 0x9e3779b9u, (lane & 15u) + 1u);
}

__global__
__launch_bounds__(kMaximumThreads) void reused_lds_gemm_pipeline_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[kMaximumLanes];

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0xc1000000u | tid;

  if (wave == 0u)
    consan_device_test::store_lds_word(&tile[lane], stage_zero_value(lane));
  __syncthreads();

  uint32_t first = 0u;
  if (wave == 1u) {
    first = consan_device_test::load_lds_word(&tile[lane]);
    state->stage_zero[lane] = first;
  }

  // Qwen's generated GEMM uses this edge to retire the first tile's readers
  // before the producer wave overwrites the same LDS addresses. Publication
  // of the replacement tile below remains present in both test members.
  if (state->synchronize_reuse != 0u)
    __syncthreads();

  if (wave == 0u)
    consan_device_test::store_lds_word(&tile[lane], stage_one_value(lane));
  __syncthreads();

  if (wave == 1u) {
    const uint32_t second = consan_device_test::load_lds_word(&tile[lane]);
    state->stage_one[lane] = second;
    state->fused[lane] = fuse_stages(first, second, lane);
  }
}

void run_reused_lds_gemm_pipeline(bool synchronize_reuse) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize_reuse = synchronize_reuse ? 1u : 0u;
  state.wave_size = wave_size;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  reused_lds_gemm_pipeline_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0xc1000000u | tid) << "thread=" << tid;

  if (synchronize_reuse) {
    for (uint32_t lane = 0u; lane < wave_size; ++lane) {
      const uint32_t first = stage_zero_value(lane);
      const uint32_t second = stage_one_value(lane);
      EXPECT_EQ(state.stage_zero[lane], first) << "lane=" << lane;
      EXPECT_EQ(state.stage_one[lane], second) << "lane=" << lane;
      EXPECT_EQ(state.fused[lane], fuse_stages(first, second, lane)) << "lane=" << lane;
    }
  }
}

TEST(ConSanDeviceReusedLdsGemmPipelineTest, Correct) { run_reused_lds_gemm_pipeline(true); }

TEST(ConSanDeviceReusedLdsGemmPipelineTest, Incorrect) { run_reused_lds_gemm_pipeline(false); }

} // namespace
