/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_bf16.h>

static __global__ void bf16_atomic_add_kernel(__hip_bfloat16* result, float* values, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(result, __float2bfloat16(values[i]));
  }
}

static __global__ void bf162_atomic_add_kernel(__hip_bfloat162* result, float2* values,
                                               size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(result, __float22bfloat162_rn(values[i]));
  }
}

HIP_TEST_CASE(Unit_bf16_atomic_add) {
  SECTION("bf16 atomic add") {
    constexpr size_t num_threads = 32;
    constexpr size_t num_blocks = 4;
    constexpr size_t total_threads = num_threads * num_blocks;

    std::vector<float> h_values(total_threads);
    for (size_t i = 0; i < total_threads; i++) {
      h_values[i] = 1.0f;  // Each thread adds 1.0
    }

    float* d_values;
    __hip_bfloat16* d_result;
    HIP_CHECK(hipMalloc(&d_values, sizeof(float) * total_threads));
    HIP_CHECK(hipMalloc(&d_result, sizeof(__hip_bfloat16)));

    HIP_CHECK(
        hipMemcpy(d_values, h_values.data(), sizeof(float) * total_threads, hipMemcpyHostToDevice));

    __hip_bfloat16 zero = __float2bfloat16(0.0f);
    HIP_CHECK(hipMemcpy(d_result, &zero, sizeof(__hip_bfloat16), hipMemcpyHostToDevice));

    bf16_atomic_add_kernel<<<num_blocks, num_threads>>>(d_result, d_values, total_threads);
    HIP_CHECK(hipDeviceSynchronize());

    __hip_bfloat16 h_result;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));

    float expected = static_cast<float>(total_threads);
    float actual = __bfloat162float(h_result);

    INFO("Expected: " << expected << " Actual: " << actual);
    // Allow small error due to bfloat16 precision
    REQUIRE(std::fabs(actual - expected) / expected < 0.01f);

    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_result));
  }

  SECTION("bf162 atomic add") {
    constexpr size_t num_threads = 32;
    constexpr size_t num_blocks = 4;
    constexpr size_t total_threads = num_threads * num_blocks;

    std::vector<float2> h_values(total_threads);
    for (size_t i = 0; i < total_threads; i++) {
      h_values[i] = float2{1.0f, 2.0f};  // Each thread adds (1.0, 2.0)
    }

    float2* d_values;
    __hip_bfloat162* d_result;
    HIP_CHECK(hipMalloc(&d_values, sizeof(float2) * total_threads));
    HIP_CHECK(hipMalloc(&d_result, sizeof(__hip_bfloat162)));

    HIP_CHECK(hipMemcpy(d_values, h_values.data(), sizeof(float2) * total_threads,
                        hipMemcpyHostToDevice));

    __hip_bfloat162 zero = __float22bfloat162_rn(float2{0.0f, 0.0f});
    HIP_CHECK(hipMemcpy(d_result, &zero, sizeof(__hip_bfloat162), hipMemcpyHostToDevice));

    bf162_atomic_add_kernel<<<num_blocks, num_threads>>>(d_result, d_values, total_threads);
    HIP_CHECK(hipDeviceSynchronize());

    __hip_bfloat162 h_result;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(__hip_bfloat162), hipMemcpyDeviceToHost));

    float2 expected =
        float2{static_cast<float>(total_threads), static_cast<float>(total_threads * 2)};
    float2 actual = __bfloat1622float2(h_result);

    INFO("Expected: (" << expected.x << ", " << expected.y << ") Actual: (" << actual.x << ", "
                       << actual.y << ")");
    REQUIRE(std::fabs(actual.x - expected.x) / expected.x < 0.01f);
    REQUIRE(std::fabs(actual.y - expected.y) / expected.y < 0.01f);

    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_result));
  }
}
