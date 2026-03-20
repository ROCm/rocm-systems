/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

HIP_TEST_CASE(Unit_hipStreamPerThread_EventRecord) {
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipEventRecord(event, hipStreamPerThread));
  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipEventDestroy(event));
}

__global__ void update_even_odd(unsigned int N, int* out) {
  for (unsigned int i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      out[i] = 2;
    } else {
      out[i] = 3;
    }
  }
}
HIP_TEST_CASE(Unit_hipStreamPerThread_EventSynchronize) {
  int* A_h = nullptr;
  int* A_d = nullptr;
  unsigned int size = 1000;

  HIP_CHECK(hipHostMalloc(&A_h, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&A_d, size * sizeof(int)));

  hipEvent_t start, end;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&end));

  HIP_CHECK(hipEventRecord(start, hipStreamPerThread));
  update_even_odd<<<1, 1>>>(size, A_d);
  HIP_CHECK(hipEventRecord(end, hipStreamPerThread));

  HIP_CHECK(hipEventSynchronize(end));
  HIP_CHECK(hipMemcpy(A_h, A_d, size * sizeof(int), hipMemcpyDeviceToHost));

  // Verify result
  for (unsigned int i = 0; i < size; ++i) {
    if (i % 2 == 0 && A_h[i] != 2)
      REQUIRE(false);
    else if (i % 2 != 0 && A_h[i] != 3) {
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(end));
}
