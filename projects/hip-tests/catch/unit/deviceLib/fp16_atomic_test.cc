/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_fp16.h>

#include <cmath>
#include <hip_test_common.hh>

static __global__ void fp16_atomic_add_kernel(__half* result, float* values, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(result, __float2half_rn(values[i]));
  }
}

static __global__ void fp162_atomic_add_kernel(__half2* result, float2* values, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(result, __float22half2_rn(values[i]));
  }
}

HIP_TEST_CASE(Unit_fp16_atomic_add) {
  SECTION("fp16 atomic add") {
    constexpr size_t num_threads = 256;
    constexpr size_t num_blocks = 4;
    constexpr size_t total_threads = num_threads * num_blocks;

    std::vector<float> h_values(total_threads);
    for (size_t i = 0; i < total_threads; i++) {
      h_values[i] = 1.0f;  // Each thread adds 1.0
    }

    float* d_values;
    __half* d_result;
    HIP_CHECK(hipMalloc(&d_values, sizeof(float) * total_threads));
    HIP_CHECK(hipMalloc(&d_result, sizeof(__half)));

    HIP_CHECK(
        hipMemcpy(d_values, h_values.data(), sizeof(float) * total_threads, hipMemcpyHostToDevice));

    __half zero = __float2half_rn(0.0f);
    HIP_CHECK(hipMemcpy(d_result, &zero, sizeof(__half), hipMemcpyHostToDevice));

    fp16_atomic_add_kernel<<<num_blocks, num_threads>>>(d_result, d_values, total_threads);
    HIP_CHECK(hipDeviceSynchronize());

    __half h_result;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(__half), hipMemcpyDeviceToHost));

    float expected = static_cast<float>(total_threads);
    float actual = __half2float(h_result);

    INFO("Expected: " << expected << " Actual: " << actual);
    // Allow small error due to fp16 precision (10 bits mantissa)
    REQUIRE(std::fabs(actual - expected) / expected < 0.01f);

    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_result));
  }

  SECTION("fp162 atomic add") {
    constexpr size_t num_threads = 256;
    constexpr size_t num_blocks = 4;
    constexpr size_t total_threads = num_threads * num_blocks;

    std::vector<float2> h_values(total_threads);
    for (size_t i = 0; i < total_threads; i++) {
      h_values[i] = float2{1.0f, 2.0f};  // Each thread adds (1.0, 2.0)
    }

    float2* d_values;
    __half2* d_result;
    HIP_CHECK(hipMalloc(&d_values, sizeof(float2) * total_threads));
    HIP_CHECK(hipMalloc(&d_result, sizeof(__half2)));

    HIP_CHECK(hipMemcpy(d_values, h_values.data(), sizeof(float2) * total_threads,
                        hipMemcpyHostToDevice));

    __half2 zero = __float22half2_rn(float2{0.0f, 0.0f});
    HIP_CHECK(hipMemcpy(d_result, &zero, sizeof(__half2), hipMemcpyHostToDevice));

    fp162_atomic_add_kernel<<<num_blocks, num_threads>>>(d_result, d_values, total_threads);
    HIP_CHECK(hipDeviceSynchronize());

    __half2 h_result;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(__half2), hipMemcpyDeviceToHost));

    float2 expected =
        float2{static_cast<float>(total_threads), static_cast<float>(total_threads * 2)};
    float2 actual = __half22float2(h_result);

    INFO("Expected: (" << expected.x << ", " << expected.y << ") Actual: (" << actual.x << ", "
                       << actual.y << ")");
    REQUIRE(std::fabs(actual.x - expected.x) / expected.x < 0.01f);
    REQUIRE(std::fabs(actual.y - expected.y) / expected.y < 0.01f);

    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_result));
  }

  SECTION("fp16 atomic add - concurrent stress test") {
    constexpr size_t num_threads = 256;
    constexpr size_t num_blocks = 16;
    constexpr size_t total_threads = num_threads * num_blocks;

    std::vector<float> h_values(total_threads);
    float sum_expected = 0.0f;
    for (size_t i = 0; i < total_threads; i++) {
      h_values[i] = static_cast<float>(i % 10) * 0.5f;
      const __half rounded_value = __float2half_rn(h_values[i]);
      sum_expected += __half2float(rounded_value);
    }

    float* d_values;
    __half* d_result;
    HIP_CHECK(hipMalloc(&d_values, sizeof(float) * total_threads));
    HIP_CHECK(hipMalloc(&d_result, sizeof(__half)));

    HIP_CHECK(
        hipMemcpy(d_values, h_values.data(), sizeof(float) * total_threads, hipMemcpyHostToDevice));

    __half zero = __float2half_rn(0.0f);
    HIP_CHECK(hipMemcpy(d_result, &zero, sizeof(__half), hipMemcpyHostToDevice));

    fp16_atomic_add_kernel<<<num_blocks, num_threads>>>(d_result, d_values, total_threads);
    HIP_CHECK(hipDeviceSynchronize());

    __half h_result;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(__half), hipMemcpyDeviceToHost));

    float actual = __half2float(h_result);

    INFO("Expected: " << sum_expected << " Actual: " << actual);
    // Higher tolerance for large sums due to accumulated rounding
    REQUIRE(std::fabs(actual - sum_expected) / sum_expected < 0.08f);

    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_result));
  }
}
