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
__managed__ int m_pa_before = 0;
__managed__ int m_pa_after = 0;

static __global__ void managed_add(size_t size) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    m_B[i] += m_A[i];
  }
}

static __global__ void managed_inc() { atomicAdd(&m_X, 1.0f); }

static __global__ void managed_touch(int* p) { (void)*p; }

HIP_TEST_CASE(Unit_hipManagedKeyword_SingleGpu) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    int managed_memory = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, i));
    if (!managed_memory) {
      HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported);
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
      HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported);
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

HIP_TEST_CASE(Unit_hipManagedKeyword_hipPointerGetAttributes_BeforeKernel) {
  CHECK_MANAGED_MEMORY_SUPPORT

  hipPointerAttribute_t attrs{};
  HIP_CHECK(hipPointerGetAttributes(&attrs, &m_pa_before));
  REQUIRE(attrs.type == hipMemoryTypeManaged);
  REQUIRE(attrs.isManaged == true);
  REQUIRE(attrs.hostPointer != nullptr);
  REQUIRE(attrs.devicePointer != nullptr);
  REQUIRE(attrs.hostPointer == attrs.devicePointer);
}

HIP_TEST_CASE(Unit_hipManagedKeyword_hipPointerGetAttributes_AfterKernel) {
  CHECK_MANAGED_MEMORY_SUPPORT

  managed_touch<<<1, 1>>>(&m_pa_after);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipPointerAttribute_t attrs{};
  HIP_CHECK(hipPointerGetAttributes(&attrs, &m_pa_after));
  REQUIRE(attrs.type == hipMemoryTypeManaged);
  REQUIRE(attrs.isManaged == true);
  REQUIRE(attrs.hostPointer != nullptr);
  REQUIRE(attrs.devicePointer != nullptr);
  REQUIRE(attrs.hostPointer == attrs.devicePointer);
}
