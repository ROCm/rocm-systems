// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// CDNA5 tensor-descriptor staging distilled from the PyTorch/Triton
/// tensor-descriptor add workload. One wave transfers two input tiles into
/// LDS with TENSOR_LOAD_TO_LDS, a peer wave consumes and adds them, and a
/// third wave drains the result with TENSOR_STORE_FROM_LDS. The adjacent pair
/// differs only in the first workgroup publication barrier.

#include "consan_device_test_support.h"

#include <array>

namespace {

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 3u * kWaveSize;
constexpr uint32_t kCanary = 0xccccccccu;

using I32x4 = __attribute__((ext_vector_type(4))) int32_t;
using I32x8 = __attribute__((ext_vector_type(8))) int32_t;

struct DeviceState {
  uint32_t synchronize;
  uint32_t output[kWaveSize + 2u];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ I32x4 tensor_group0(const uint32_t *global_base,
                                               const uint32_t *lds_base) {
  const uintptr_t global_address = reinterpret_cast<uintptr_t>(global_base);
  const uintptr_t lds_address = reinterpret_cast<uintptr_t>(lds_base);
  return I32x4{1, static_cast<int32_t>(lds_address), static_cast<int32_t>(global_address),
               static_cast<int32_t>(((global_address >> 32u) & 0x01ffffffu) | 0x80000000u)};
}

__device__ __forceinline__ I32x8 tensor_group1() {
  // One rank of 32 dwords: element-size code 2, tensor extent 32, tile extent
  // 32. Higher-dimensional descriptor groups are zero below.
  return I32x8{static_cast<int32_t>(2u << 16u),
               static_cast<int32_t>(kWaveSize << 16u),
               0,
               static_cast<int32_t>(kWaveSize << 16u),
               0,
               0,
               0,
               0};
}

__device__ __forceinline__ void tensor_load(const uint32_t *global_base, uint32_t *lds_base) {
  const I32x4 empty4{};
  const I32x8 empty8{};
  __builtin_amdgcn_tensor_load_to_lds(tensor_group0(global_base, lds_base), tensor_group1(), empty4,
                                      empty4, empty8, 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);
}

__device__ __forceinline__ void tensor_store(uint32_t *global_base, const uint32_t *lds_base) {
  const I32x4 empty4{};
  const I32x8 empty8{};
  __builtin_amdgcn_tensor_store_from_lds(tensor_group0(global_base, lds_base), tensor_group1(),
                                         empty4, empty4, empty8, 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);
}

__global__ __launch_bounds__(kThreadCount) void gfx1250_tensor_descriptor_pipeline_for_consan(
    const uint32_t *input_a, const uint32_t *input_b, DeviceState *state) {
  __shared__ uint32_t staged_a[kWaveSize];
  __shared__ uint32_t staged_b[kWaveSize];
  __shared__ uint32_t staged_output[kWaveSize];
  __shared__ uint32_t publication[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);

  if (tid < kWaveSize) {
    tensor_load(input_a, staged_a);
    tensor_load(input_b, staged_b);
    publication[lane] = 0x1250d000u | lane;
  }
  if (state->synchronize != 0u)
    __syncthreads();

  if (tid >= kWaveSize && tid < 2u * kWaveSize) {
    const uint32_t published = consan_device_test::load_lds_word(&publication[lane]);
    const uint32_t lhs = consan_device_test::load_lds_word(&staged_a[lane]);
    const uint32_t rhs = consan_device_test::load_lds_word(&staged_b[lane]);
    staged_output[lane] = lhs + rhs + (published == (0x1250d000u | lane) ? 0u : 1u);
  }
  __syncthreads();

  if (tid >= 2u * kWaveSize)
    tensor_store(&state->output[1], staged_output);
  state->control[tid] = 0x1250e000u | tid;
}

void run_gfx1250_tensor_descriptor_pipeline(bool synchronize) {
  std::array<uint32_t, kWaveSize> input_a{};
  std::array<uint32_t, kWaveSize> input_b{};
  for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
    input_a[lane] = 0x10000000u + lane * 29u;
    input_b[lane] = 0x01000000u + lane * 17u + 3u;
  }

  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  initial.output[0] = kCanary;
  initial.output[kWaveSize + 1u] = kCanary;
  for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
    initial.output[lane + 1u] = kCanary;

  uint32_t *device_a = nullptr;
  uint32_t *device_b = nullptr;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_a, sizeof(input_a)));
  HIP_ASSERT(hipMalloc(&device_b, sizeof(input_b)));
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_a, input_a.data(), sizeof(input_a), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(device_b, input_b.data(), sizeof(input_b), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  gfx1250_tensor_descriptor_pipeline_for_consan<<<1, kThreadCount>>>(device_a, device_b,
                                                                     device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  HIP_ASSERT(hipFree(device_b));
  HIP_ASSERT(hipFree(device_a));

  for (uint32_t tid = 0u; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0x1250e000u | tid) << "thread=" << tid;
  EXPECT_EQ(result.output[0], kCanary);
  EXPECT_EQ(result.output[kWaveSize + 1u], kCanary);
  if (synchronize) {
    for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
      EXPECT_EQ(result.output[lane + 1u], input_a[lane] + input_b[lane]) << "lane=" << lane;
  }
}

TEST(ConSanDeviceGfx1250TensorDescriptorPipelineTest, Correct) {
  run_gfx1250_tensor_descriptor_pipeline(true);
}

TEST(ConSanDeviceGfx1250TensorDescriptorPipelineTest, Incorrect) {
  run_gfx1250_tensor_descriptor_pipeline(false);
}

} // namespace
