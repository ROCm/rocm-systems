// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// LDS histogram and collision-heavy global scatter distilled from PyTorch.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kBinCount = 8u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t published_wave_total;
  uint32_t histogram[kBinCount];
  uint32_t scatter[kBinCount];
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void histogram_scatter_for_consan(DeviceState *state) {
  __shared__ uint32_t bins[kBinCount];
  __shared__ uint32_t wave_total;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  if (tid < kBinCount)
    bins[tid] = 0u;
  __syncthreads();

  const uint32_t bin = (5u * tid + 3u) & (kBinCount - 1u);
  atomicAdd(&bins[bin], 1u);
  (void)__hip_atomic_fetch_add(&state->scatter[bin], tid + 1u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
  state->control[tid] = 0x79000000u | tid;
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");

  // Histogram kernels commonly publish a per-wave completion total before a
  // finalizer consumes the combined bins. Keep that ordinary LDS handoff
  // alongside the collision-heavy LDS and global atomics so every target has
  // a deterministic behavioral race even when its prototype does not yet
  // model a particular atomic encoding.
  if (wave == 1u && lane == 0u) {
    wave_total = warpSize;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }

  // The scatter atomics remain exact in both members and form an independent
  // oracle. Only publication of the LDS histogram to its ordinary readers is
  // broken in the incorrect member.
  if (state->synchronize)
    __syncthreads();
  if (tid == 0u) {
    state->published_wave_total = wave_total;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  if (tid < kBinCount) {
    state->histogram[tid] = bins[tid];
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

void run_histogram_scatter(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  histogram_scatter_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  uint32_t expected_histogram[kBinCount] = {};
  uint32_t expected_scatter[kBinCount] = {};
  for (uint32_t tid = 0; tid < thread_count; ++tid) {
    const uint32_t bin = (5u * tid + 3u) & (kBinCount - 1u);
    ++expected_histogram[bin];
    expected_scatter[bin] += tid + 1u;
    EXPECT_EQ(state.control[tid], 0x79000000u | tid) << "thread=" << tid;
  }
  EXPECT_EQ(state.wave_size, wave_size);
  if (synchronize)
    EXPECT_EQ(state.published_wave_total, wave_size);
  for (uint32_t bin = 0; bin < kBinCount; ++bin) {
    EXPECT_EQ(state.scatter[bin], expected_scatter[bin]) << "bin=" << bin;
    if (synchronize)
      EXPECT_EQ(state.histogram[bin], expected_histogram[bin]) << "bin=" << bin;
  }
}

TEST(ConSanDeviceHistogramScatterTest, Correct) { run_histogram_scatter(true); }

TEST(ConSanDeviceHistogramScatterTest, Incorrect) { run_histogram_scatter(false); }

} // namespace
