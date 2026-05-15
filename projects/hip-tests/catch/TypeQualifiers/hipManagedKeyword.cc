/*
   Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
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

TEST_CASE("Unit_hipManagedKeyword_SingleGpu") {
  for (size_t i = 0; i < N; i++) {
    m_A[i] = 1.0f;
    m_B[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = N / blockSize;

  managed_add<<<numBlocks, blockSize>>>(N);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());

  float maxError = 0.0f;
  for (size_t i = 0; i < N; i++) {
    INFO("Reading output from managed variable: Index: " << i << " output: " << m_B[i]);
    REQUIRE(3.0f == m_B[i]);
  }
}

TEST_CASE("Unit_hipManagedKeyword_MultiGpu") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; i++) {
    int managed_memory = 0;
    HIPCHECK(hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, i));
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
