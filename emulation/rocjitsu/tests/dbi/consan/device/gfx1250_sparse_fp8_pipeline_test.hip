// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Sparse FP8 matrix staging distilled from Tensile's gfx1250 spmm_f8_ml
/// workload: byte and D16-high LDS stores, byte and transposed metadata loads,
/// split workgroup synchronization, and live SWMMAC state.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t head_canary;
  uint32_t packed_bytes[kWaveSize];
  uint32_t matrix_bits[kWaveSize];
  uint32_t tail_canary;
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_sparse_bytes(uint32_t address, uint32_t value) {
  asm volatile("ds_store_b8 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "ds_store_b8_d16_hi %0, %1 offset:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_wait_dscnt 0\n\t"
               :
               : "v"(address), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_sparse_bytes(uint32_t address) {
  uint32_t low = 0u;
  uint32_t high = 0u;
  asm volatile("ds_load_u8 %0, %2\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "ds_load_u8 %1, %2 offset:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               : "=&v"(low), "=&v"(high)
               : "v"(address)
               : "memory");
  return low | (high << 8u);
}

__device__ __forceinline__ uint32_t run_unit_sparse_fp8(uint32_t metadata_address) {
  uint32_t result = 0u;
  asm volatile("v_mov_b32 v0, 0\n\t"
               "v_mov_b32 v1, 0\n\t"
               "v_mov_b32 v2, 0\n\t"
               "v_mov_b32 v3, 0\n\t"
               "v_mov_b32 v4, 0\n\t"
               "v_mov_b32 v5, 0\n\t"
               "v_mov_b32 v6, 0\n\t"
               "v_mov_b32 v7, 0\n\t"
               "v_mov_b32 v20, 0x38383838\n\t"
               "v_mov_b32 v21, 0x38383838\n\t"
               "v_mov_b32 v22, 0x38383838\n\t"
               "v_mov_b32 v23, 0x38383838\n\t"
               "v_mov_b32 v24, 0x38383838\n\t"
               "v_mov_b32 v25, 0x38383838\n\t"
               "v_mov_b32 v26, 0x38383838\n\t"
               "v_mov_b32 v27, 0x38383838\n\t"
               "v_mov_b32 v54, 0x38383838\n\t"
               "v_mov_b32 v55, 0x38383838\n\t"
               "v_mov_b32 v56, 0x38383838\n\t"
               "v_mov_b32 v57, 0x38383838\n\t"
               "v_mov_b32 v58, 0x38383838\n\t"
               "v_mov_b32 v59, 0x38383838\n\t"
               "v_mov_b32 v60, 0x38383838\n\t"
               "v_mov_b32 v61, 0x38383838\n\t"
               "v_mov_b32 v62, 0x38383838\n\t"
               "v_mov_b32 v63, 0x38383838\n\t"
               "v_mov_b32 v64, 0x38383838\n\t"
               "v_mov_b32 v65, 0x38383838\n\t"
               "v_mov_b32 v66, 0x38383838\n\t"
               "v_mov_b32 v67, 0x38383838\n\t"
               "v_mov_b32 v68, 0x38383838\n\t"
               "v_mov_b32 v69, 0x38383838\n\t"
               "ds_load_tr8_b64 v[122:123], %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               "v_swmmac_f32_16x16x128_fp8_fp8 v[0:7], v[20:27], v[54:69], v[122:123]\n\t"
               "s_nop 7\n\t"
               "v_mov_b32 %0, v0\n\t"
               : "=v"(result)
               : "v"(metadata_address)
               : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20", "v21", "v22", "v23", "v24",
                 "v25", "v26", "v27", "v54", "v55", "v56", "v57", "v58", "v59", "v60", "v61", "v62",
                 "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v122", "v123", "memory");
  return result;
}

__global__
__launch_bounds__(kThreadCount) void gfx1250_sparse_fp8_pipeline_for_consan(DeviceState *state) {
  __shared__ __align__(8) uint8_t packed_bytes[2u * kWaveSize];
  __shared__ __align__(8) uint64_t sparse_metadata[kWaveSize];

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  if (tid < kWaveSize) {
    const uint32_t byte_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&packed_bytes[2u * lane]) & 0xffffu);
    // Keep both bytes of the D16-high half identical. The store contract is
    // about selecting that half rather than which byte packing a producer
    // happened to use before the instruction.
    store_sparse_bytes(byte_address, 0x5a5a00a5u);
    // Every two-bit selector pair is {0, 1}. With packed FP8 1.0 operands,
    // each sparse dot product has 64 selected terms and must equal 64.0f.
    sparse_metadata[lane] = 0x4444444444444444ull;
  }

  if (state->synchronize != 0u)
    __syncthreads();

  if (tid >= kWaveSize) {
    const uint32_t byte_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&packed_bytes[2u * lane]) & 0xffffu);
    const uint32_t metadata_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sparse_metadata[lane]) & 0xffffu);
    state->packed_bytes[lane] = load_sparse_bytes(byte_address);
    state->matrix_bits[lane] = run_unit_sparse_fp8(metadata_address);
  }
  state->control[tid] = 0x12500000u | tid;
}

void run_gfx1250_sparse_fp8_pipeline(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  initial.head_canary = 0x13579bdfu;
  initial.tail_canary = 0x2468ace0u;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  gfx1250_sparse_fp8_pipeline_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.head_canary, 0x13579bdfu);
  EXPECT_EQ(result.tail_canary, 0x2468ace0u);
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0x12500000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    EXPECT_EQ(result.packed_bytes[lane], 0x5aa5u) << "lane=" << lane;
    EXPECT_EQ(result.matrix_bits[lane], 0x42800000u) << "lane=" << lane;
  }
}

TEST(ConSanDeviceGfx1250SparseFp8PipelineTest, Correct) { run_gfx1250_sparse_fp8_pipeline(true); }

TEST(ConSanDeviceGfx1250SparseFp8PipelineTest, Incorrect) {
  run_gfx1250_sparse_fp8_pipeline(false);
}

} // namespace
