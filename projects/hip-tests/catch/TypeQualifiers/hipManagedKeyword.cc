/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
   This testcase verifies the hipManagedKeyword basic scenario
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define N 1048576
__managed__ float m_A[N];  // Accessible by ALL CPU and GPU functions !!!
__managed__ float m_B[N];
__managed__ int m_X = 0;

static __global__ void managed_add(size_t size) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    m_B[i] += m_A[i];
  }
}

static __global__ void managed_inc() { atomicAdd(&m_X, 1.0f); }

HIP_TEST_CASE(Unit_hipManagedKeyword_SingleGpu) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    int managed_memory = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, i));
    if (!managed_memory) {
      HipTest::HIP_SKIP_TEST("managed memory access not supported on device");
      return;
    }
  }

  for (size_t i = 0; i < N; i++) {
    m_A[i] = 1.0f;
    m_B[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = N / blockSize;

  managed_add<<<numBlocks, blockSize>>>(N);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());

  for (size_t i = 0; i < N; i++) {
    INFO("Reading output from managed variable: Index: " << i << " output: " << m_B[i]);
    REQUIRE(3.0f == m_B[i]);
  }
}

HIP_TEST_CASE(Unit_hipManagedKeyword_MultiGpu) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; i++) {
    int managed_memory = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, i));
    if (!managed_memory) {
      HipTest::HIP_SKIP_TEST("managed memory access not supported on device");
      return;
    }
  }

  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));
    managed_inc<<<1, 1>>>();
    HIP_CHECK(hipDeviceSynchronize());
  }

  INFO("Inc counter should match the device count: " << m_X << " Device count: " << numDevices);
  REQUIRE(m_X == numDevices);
}
