// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// LDS histogram and collision-heavy global scatter distilled from PyTorch.

#include "consan_device_test_support.h"

#include <hip/hip_bf16.h>

namespace {

constexpr uint32_t kBinCount = 8u;
constexpr uint32_t kPackedBf16BinCount = 4u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t published_wave_total;
  uint32_t histogram[kBinCount];
  float fp32_histogram[kBinCount];
  double fp64_histogram[kBinCount];
  uint32_t scatter[kBinCount];
  alignas(uint32_t) __hip_bfloat16 bf16_scatter[kBinCount];
  uint32_t packed_bf16_scatter[kPackedBf16BinCount];
  uint32_t packed_bf16_executed;
  uint32_t control[128];
};

__device__ __forceinline__ void relaxed_native_packed_bf16_add(uint32_t *address,
                                                               uint32_t packed_value) {
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("global_atomic_pk_add_bf16 %0, %1, off\n\t"
               "s_waitcnt vmcnt(0)\n\t"
               :
               : "v"(address), "v"(packed_value)
               : "memory");
#elif defined(__gfx1201__) || defined(__gfx1250__)
  asm volatile("global_atomic_pk_add_bf16 %0, %1, off scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               :
               : "v"(address), "v"(packed_value)
               : "memory");
#else
  (void)address;
  (void)packed_value;
#endif
}

__global__ __launch_bounds__(128) void histogram_scatter_for_consan(DeviceState *state) {
  __shared__ uint32_t bins[kBinCount];
  __shared__ float fp32_bins[kBinCount];
  __shared__ double fp64_bins[kBinCount];
  __shared__ uint32_t wave_total;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  if (tid < kBinCount) {
    bins[tid] = 0u;
    fp32_bins[tid] = 0.0f;
    fp64_bins[tid] = 0.0;
  }
  __syncthreads();

  const uint32_t bin = (5u * tid + 3u) & (kBinCount - 1u);
  atomicAdd(&bins[bin], 1u);
  atomicAdd(&fp32_bins[bin], 1.0f);
  // PyTorch histc's gfx950 FP64 specialization uses this native
  // collision-heavy ds_add_f64 shape.  CDNA3/CDNA5 lower it natively too;
  // RDNA3/RDNA4 transport the same exact arithmetic contract through their
  // compiler-selected 64-bit LDS compare/exchange loop.
  atomicAdd(&fp64_bins[bin], 1.0);
  (void)__hip_atomic_fetch_add(&state->scatter[bin], tid + 1u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
  // PyTorch-style scatter accumulation is an arithmetic collision, not a
  // publication edge.  Keep that distinction visible in the source: the HIP
  // scalar BF16 atomicAdd wrapper requests acquire-release ordering, whereas
  // the workload idiom only requires a relaxed device-scope RMW.
  (void)__scoped_atomic_fetch_add(reinterpret_cast<__bf16 *>(&state->bf16_scatter[bin]),
                                  static_cast<__bf16>(1.0f), __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE);
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1201__) || defined(__gfx1250__)
  relaxed_native_packed_bf16_add(&state->packed_bf16_scatter[tid % kPackedBf16BinCount],
                                 0x40003f80u);
  if (tid == 0u)
    state->packed_bf16_executed = 1u;
#endif
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
    state->fp32_histogram[tid] = fp32_bins[tid];
    state->fp64_histogram[tid] = fp64_bins[tid];
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
    EXPECT_EQ(__bfloat162float(state.bf16_scatter[bin]),
              static_cast<float>(expected_histogram[bin]))
        << "bin=" << bin;
    if (synchronize) {
      EXPECT_EQ(state.histogram[bin], expected_histogram[bin]) << "bin=" << bin;
      EXPECT_EQ(state.fp32_histogram[bin], static_cast<float>(expected_histogram[bin]))
          << "bin=" << bin;
      EXPECT_EQ(state.fp64_histogram[bin], static_cast<double>(expected_histogram[bin]))
          << "bin=" << bin;
    }
  }
  if (state.packed_bf16_executed != 0u) {
    EXPECT_EQ(state.packed_bf16_executed, 1u);
    const uint32_t additions = thread_count / kPackedBf16BinCount;
    const uint32_t expected =
        static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(additions))) |
        (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(2u * additions))) << 16u);
    for (uint32_t bin = 0; bin < kPackedBf16BinCount; ++bin)
      EXPECT_EQ(state.packed_bf16_scatter[bin], expected) << "bin=" << bin;
  } else {
    for (uint32_t bin = 0; bin < kPackedBf16BinCount; ++bin)
      EXPECT_EQ(state.packed_bf16_scatter[bin], 0u) << "bin=" << bin;
  }
}

TEST(ConSanDeviceHistogramScatterTest, Correct) { run_histogram_scatter(true); }

TEST(ConSanDeviceHistogramScatterTest, Incorrect) { run_histogram_scatter(false); }

} // namespace
