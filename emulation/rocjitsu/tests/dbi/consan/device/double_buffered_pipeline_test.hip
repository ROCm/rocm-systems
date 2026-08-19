// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Double-buffered wide-LDS staging distilled from attention and GEMM kernels.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t output[64];
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void double_buffered_pipeline_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint4 tiles[2][64];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x78000000u | tid;

  if (wave == 0u) {
    tiles[0][lane] =
        make_uint4(0x1000u + lane, 0x2000u + 3u * lane, 0x3000u + 5u * lane, 0x4000u + 7u * lane);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  __syncthreads();

  uint32_t first = 0u;
  if (wave == 1u) {
    const uint4 value = tiles[0][lane];
    first = value.x + value.y + value.z + value.w;
    first ^= __shfl_xor(first, 1u, warpSize);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  // Every consumer has finished with stage zero before its storage is reused.
  __syncthreads();

  if (wave == 0u) {
    tiles[1][lane] = make_uint4(0x5000u + 11u * lane, 0x6000u + 13u * lane, 0x7000u + 17u * lane,
                                0x8000u + 19u * lane);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  // The incorrect member removes only this stage-publication edge. Both
  // stages retain the same vector traffic, lane shuffle, and control shape.
  if (state->synchronize)
    __syncthreads();

  if (wave == 1u) {
    const uint4 value = tiles[1][lane];
    const uint32_t second = value.x + value.y + value.z + value.w;
    const uint32_t epilogue = consan_device_test::fused_epilogue_mix(state->control, lane, 127u);
    state->output[lane] = first ^ second ^ epilogue;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

void run_double_buffered_pipeline(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  double_buffered_pipeline_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x78000000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      const uint32_t peer = lane ^ 1u;
      const uint32_t lane_first = 0xa000u + 16u * lane;
      const uint32_t peer_first = 0xa000u + 16u * peer;
      const uint32_t second = 0x1a000u + 60u * lane;
      const uint32_t epilogue =
          consan_device_test::fused_epilogue_mix_reference(state.control, lane, 127u);
      EXPECT_EQ(state.output[lane], (lane_first ^ peer_first) ^ second ^ epilogue)
          << "lane=" << lane;
    }
  }
}

TEST(ConSanDeviceDoubleBufferedPipelineTest, Correct) { run_double_buffered_pipeline(true); }

TEST(ConSanDeviceDoubleBufferedPipelineTest, Incorrect) { run_double_buffered_pipeline(false); }

} // namespace
