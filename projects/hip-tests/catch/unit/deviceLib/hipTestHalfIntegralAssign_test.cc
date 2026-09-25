/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Regression test for __half::operator=(T) integral assignment.
 *
 * The integral assignment operator template was previously __device__-only,
 * while the matching constructor was __HOST_DEVICE__. This caused compilation
 * failures in any __host__ __device__ function using `half_var = int_val;`.
 *
 * These tests verify that integral assignment works in host, device, and
 * __host__ __device__ contexts.
 */

#include <hip/hip_fp16.h>
#include <hip_test_common.hh>

/**
 * __host__ __device__ helper that exercises integral operator= for multiple
 * integer types. Without the fix, the host pass of hipcc rejects this
 * function because operator=(T) was __device__-only.
 */
__host__ __device__ void halfIntegralAssignHelper(float* results) {
  __half h;
  int i = 3;
  h = i;
  results[0] = __half2float(h);

  short s = 7;
  h = s;
  results[1] = __half2float(h);

  unsigned u = 10;
  h = u;
  results[2] = __half2float(h);

  long long ll = 15;
  h = ll;
  results[3] = __half2float(h);
}

__global__ void halfIntegralAssignKernel(float* results) {
  halfIntegralAssignHelper(results);
}

/**
 * Test Description
 * ------------------------
 * - Calls the __host__ __device__ helper from host code.
 * - Compilation of this test is the primary regression check.
 */
HIP_TEST_CASE(Unit_hipTestHalfIntegralAssign_Host) {
  float results[4] = {0};
  halfIntegralAssignHelper(results);

  REQUIRE(results[0] == 3.0f);
  REQUIRE(results[1] == 7.0f);
  REQUIRE(results[2] == 10.0f);
  REQUIRE(results[3] == 15.0f);
}

/**
 * Test Description
 * ------------------------
 * - Launches a kernel that calls the same __host__ __device__ helper.
 * - Verifies integral assignment also works on device.
 */
HIP_TEST_CASE(Unit_hipTestHalfIntegralAssign_Device) {
  constexpr size_t numResults = 4;
  float* results_d = nullptr;
  float results_h[numResults] = {0};

  HIP_CHECK(hipMalloc(&results_d, numResults * sizeof(float)));

  halfIntegralAssignKernel<<<1, 1>>>(results_d);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(results_h, results_d, numResults * sizeof(float),
                      hipMemcpyDeviceToHost));

  REQUIRE(results_h[0] == 3.0f);
  REQUIRE(results_h[1] == 7.0f);
  REQUIRE(results_h[2] == 10.0f);
  REQUIRE(results_h[3] == 15.0f);

  HIP_CHECK(hipFree(results_d));
}
