/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

constexpr int size = 5;
constexpr int sizeBytes = size * sizeof(int);

/**
 * Kernel to perform Square of input data.
 */
static __global__ void squareKernel(int* arr) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = arr[i] * arr[i];
  arr[i] = temp;
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks the following scenario,
 *  - 1) Perform Valid Kernel Operation and check received errors
 *  - 2) Perform the Invalid Kernel Operation and check received errors
 *  - 3) Call any hip API with valid arguments and check the received errors
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastErrorOnAbort.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipGetLastError_KernelFailure_ValidAndInvalidOperations") {
  DISABLE_CORE_DUMPS();

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, sizeBytes));
  REQUIRE(devMem != nullptr);

  hipError_t ret;

  // Performing Valid operation
  squareKernel<<<1, size, 0, 0>>>(devMem);

  ret = hipGetLastError();
  REQUIRE(ret == hipSuccess);

  ret = hipDeviceSynchronize();
  REQUIRE(ret == hipSuccess);

  ret = hipGetLastError();
  REQUIRE(ret == hipSuccess);

  // Performing Invalid operation
  squareKernel<<<1, 1, 0, 0>>>(nullptr);

  ret = hipGetLastError();
  REQUIRE(ret == hipSuccess);

  ret = hipDeviceSynchronize();
  REQUIRE(ret == hipErrorIllegalAddress);

  ret = hipGetLastError();
  REQUIRE(ret == hipErrorIllegalAddress);

  // Again Performing Valid operation
  int* devMemNew = nullptr;
  ret = hipMalloc(&devMemNew, sizeBytes);
  REQUIRE(ret == hipErrorIllegalAddress);

  ret = hipGetLastError();
  REQUIRE(ret == hipErrorIllegalAddress);

  RESTORE_CORE_DUMPS();
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks the following scenario,
 *  - 1) Perform the Invalid Kernel Operation on Device 0, check received errors
 *  - 2) Perform Valid Kernel Operation on Device 1, check received errors
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastErrorOnAbort.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipGetLastError_KernelFailure_TwoDevices") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  DISABLE_CORE_DUMPS();

  // Perform Invalid operation on Device 0
  HIP_CHECK(hipSetDevice(0));

  hipError_t ret;

  squareKernel<<<1, 1, 0, 0>>>(nullptr);

  ret = hipGetLastError();
  REQUIRE(ret == hipSuccess);

  ret = hipDeviceSynchronize();
  REQUIRE(ret == hipErrorIllegalAddress);

  ret = hipGetLastError();
  REQUIRE(ret == hipErrorIllegalAddress);

  // Perform Valid operation on Device 1
  hipError_t expectedError =
#if HT_AMD
      hipErrorIllegalAddress;
#else
      hipErrorUnknown;
#endif

  ret = hipSetDevice(1);
  REQUIRE(ret == expectedError);

  int* devMem = nullptr;
  ret = hipMalloc(&devMem, sizeBytes);
  REQUIRE(ret == expectedError);

  squareKernel<<<1, size, 0, 0>>>(devMem);

  ret = hipGetLastError();
  REQUIRE(ret == expectedError);

  ret = hipDeviceSynchronize();
  REQUIRE(ret == expectedError);

  ret = hipGetLastError();
  REQUIRE(ret == expectedError);

  RESTORE_CORE_DUMPS();
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks the following scenario,
 *  - 1) Perform the Invalid Kernel Operation in Stream 1, check received errors
 *  - 2) Perform Valid Kernel Operation in Stream 2, check received errors
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastErrorOnAbort.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipGetLastError_KernelFailure_TwoStreams") {
  DISABLE_CORE_DUMPS();

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, sizeBytes));
  REQUIRE(devMem != nullptr);

  hipStream_t stream1, stream2;

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  hipError_t ret;

  // Perform Invalid operation on stream 1
  squareKernel<<<1, 1, 0, stream1>>>(nullptr);

  ret = hipGetLastError();
  REQUIRE(ret == hipSuccess);

  ret = hipDeviceSynchronize();
  REQUIRE(ret == hipErrorIllegalAddress);

  ret = hipGetLastError();
  REQUIRE(ret == hipErrorIllegalAddress);

  // Perform Invalid operation on stream 2
  squareKernel<<<1, size, 0, stream2>>>(devMem);

  ret = hipGetLastError();
  REQUIRE(ret == hipErrorIllegalAddress);

  ret = hipDeviceSynchronize();
  REQUIRE(ret == hipErrorIllegalAddress);

  ret = hipGetLastError();
  REQUIRE(ret == hipErrorIllegalAddress);

  RESTORE_CORE_DUMPS();
}
