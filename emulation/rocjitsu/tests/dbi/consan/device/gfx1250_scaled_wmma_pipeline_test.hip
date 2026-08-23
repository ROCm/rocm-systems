// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Scaled FP8 WMMA staging distilled from Tensile's gfx1250 kernels. A
/// producer wave publishes packed FP8 fragments through LDS; a consumer wave
/// loads them into the native matrix operands and checks the exact scaled-WMMA
/// result while ConSan instruments the publication edge.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kFragmentWords = 16u;
constexpr uint32_t kResultWords = 8u;
constexpr uint32_t kPackedFp8Ones = 0x38383838u;
constexpr uint32_t kExpectedResult = 0x43000000u; // 128.0f

struct DeviceState {
  uint32_t synchronize;
  uint32_t result[kWaveSize * kResultWords];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void run_scaled_wmma(uint32_t a_address, uint32_t b_address,
                                                uint32_t *result) {
  uint32_t result0;
  uint32_t result1;
  uint32_t result2;
  uint32_t result3;
  uint32_t result4;
  uint32_t result5;
  uint32_t result6;
  uint32_t result7;
  asm volatile("ds_load_b128 v[18:21], %8\n\t"
               "ds_load_b128 v[22:25], %8 offset:16\n\t"
               "ds_load_b128 v[26:29], %8 offset:32\n\t"
               "ds_load_b128 v[30:33], %8 offset:48\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "ds_load_b128 v[52:55], %9\n\t"
               "ds_load_b128 v[56:59], %9 offset:16\n\t"
               "ds_load_b128 v[60:63], %9 offset:32\n\t"
               "ds_load_b128 v[64:67], %9 offset:48\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_wait_dscnt 0\n\t"
               "v_mov_b32 v0, 0x7f7f7f7f\n\t"
               "v_mov_b32 v4, 0x7f7f7f7f\n\t"
               // v_wmma_scale_f32_16x16x128_f8f6f4 v[6:13], v[18:33],
               // v[52:67], 0, v0, v4. The installed assembler cannot spell the paired
               // encoding yet, so retain the reviewed native instruction words.
               ".long 0xcc350000\n\t"
               ".long 0x04020900\n\t"
               ".long 0xcc330006\n\t"
               ".long 0x02026912\n\t"
               "s_nop 7\n\t"
               "v_mov_b32 %0, v6\n\t"
               "v_mov_b32 %1, v7\n\t"
               "v_mov_b32 %2, v8\n\t"
               "v_mov_b32 %3, v9\n\t"
               "v_mov_b32 %4, v10\n\t"
               "v_mov_b32 %5, v11\n\t"
               "v_mov_b32 %6, v12\n\t"
               "v_mov_b32 %7, v13\n\t"
               : "=v"(result0), "=v"(result1), "=v"(result2), "=v"(result3), "=v"(result4),
                 "=v"(result5), "=v"(result6), "=v"(result7)
               : "v"(a_address), "v"(b_address)
               : "v0", "v4", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v18", "v19",
                 "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                 "v32", "v33", "v52", "v53", "v54", "v55", "v56", "v57", "v58", "v59", "v60", "v61",
                 "v62", "v63", "v64", "v65", "v66", "v67", "memory");
  result[0] = result0;
  result[1] = result1;
  result[2] = result2;
  result[3] = result3;
  result[4] = result4;
  result[5] = result5;
  result[6] = result6;
  result[7] = result7;
}

__global__
__launch_bounds__(kThreadCount) void gfx1250_scaled_wmma_pipeline_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint32_t a[kWaveSize][kFragmentWords];
  __shared__ __align__(16) uint32_t b[kWaveSize][kFragmentWords];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);

  if (tid < kWaveSize) {
#pragma unroll
    for (uint32_t word = 0; word < kFragmentWords; ++word) {
      a[lane][word] = kPackedFp8Ones;
      b[lane][word] = kPackedFp8Ones;
    }
  }
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize) {
    const uint32_t a_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&a[lane][0]) & 0xffffu);
    const uint32_t b_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&b[lane][0]) & 0xffffu);
    run_scaled_wmma(a_address, b_address, &state->result[lane * kResultWords]);
  }
  state->control[tid] = 0x5ca1e000u | tid;
}

void run_gfx1250_scaled_wmma_pipeline(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  gfx1250_scaled_wmma_pipeline_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0x5ca1e000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    for (uint32_t word = 0; word < kResultWords; ++word)
      EXPECT_EQ(result.result[lane * kResultWords + word], kExpectedResult)
          << "lane=" << lane << " result word=" << word;
}

TEST(ConSanDeviceGfx1250ScaledWmmaPipelineTest, Correct) { run_gfx1250_scaled_wmma_pipeline(true); }

TEST(ConSanDeviceGfx1250ScaledWmmaPipelineTest, Incorrect) {
  run_gfx1250_scaled_wmma_pipeline(false);
}

} // namespace
