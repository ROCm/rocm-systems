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

/**
 * Kernel to double each element in the given array
 */
static __global__ void doubleKernel(int *arr) {
  size_t i = threadIdx.x;
  arr[i] += arr[i];
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following scenario,
 *  - 1) Using hipStreamGetAttribute, get the default value of attribute
 *  -    hipStreamAttributeSynchronizationPolicy
 *  - 2) Using hipStreamSetAttribute, set the possible values of
         hipStreamAttributeSynchronizationPolicy attribute
 *  - 3) And validate it by getting value using hipStreamGetAttribute
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSetGetAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipStreamSetAttribute_hipStreamGetAttribute_Basic") {
  constexpr int N = 1024;
  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 5;
  }

  int *devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
  REQUIRE(devMem != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueToSet;
  hipSynchronizationPolicy syncPolicy =
      GENERATE(hipSynchronizationPolicy::hipSyncPolicyAuto,
               hipSynchronizationPolicy::hipSyncPolicySpin,
               hipSynchronizationPolicy::hipSyncPolicyYield,
               hipSynchronizationPolicy::hipSyncPolicyBlockingSync);
  valueToSet.syncPolicy = syncPolicy;
  HIP_CHECK(hipStreamSetAttribute(stream, attr, &valueToSet));

  hipStreamAttrValue valueOut;
  HIP_CHECK(hipStreamGetAttribute(stream, attr, &valueOut));
  REQUIRE(valueOut.syncPolicy == syncPolicy);

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem, N * sizeof(int),
                           hipMemcpyHostToDevice, stream));
  doubleKernel<<<1, N, 0, stream>>>(devMem);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, N * sizeof(int),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    INFO("At index " << i << " Expected value = 10 "
                     << " Got value = " << hostMem[i]);
    REQUIRE(hostMem[i] == 10);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following Negative scenarios
 *  - for the hipStreamGetAttribute,
 *  - 1) With Invalid stream
 *  - 2) With Invalid attribute
 *  - 3) With Invalid Attribute value
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSetGetAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipStreamGetAttribute_Negative") {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueOut;

  SECTION("With Invalid stream") {
    hipStream_t invalidStream = reinterpret_cast<hipStream_t>(-1);
    HIP_CHECK_ERROR(hipStreamGetAttribute(invalidStream, attr, &valueOut),
                    hipErrorInvalidResourceHandle);
  }

  SECTION("With invalid attribute") {
    attr = static_cast<hipStreamAttrID>(-1);
    HIP_CHECK_ERROR(hipStreamGetAttribute(stream, attr, &valueOut),
                    hipErrorInvalidValue);
  }
#if HT_AMD
  // Facing segmentation fault issue in CUDA
  SECTION("With invalid attribute value") {
    HIP_CHECK_ERROR(hipStreamGetAttribute(stream, attr, nullptr),
                    hipErrorInvalidValue);
  }
#endif
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following Negative scenarios
 *  - for the hipStreamSetAttribute,
 *  - 1) With Invalid stream
 *  - 2) With Invalid attribute
 *  - 3) With Invalid Attribute value
 *  - 4) With Invalid hipSynchronizationPolicy
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSetGetAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipStreamSetAttribute_Negative") {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueToSet;

  SECTION("With Invalid stream") {
    hipStream_t invalidStream = reinterpret_cast<hipStream_t>(-1);
    valueToSet.syncPolicy = hipSynchronizationPolicy::hipSyncPolicyAuto;
    HIP_CHECK_ERROR(hipStreamSetAttribute(invalidStream, attr, &valueToSet),
                    hipErrorInvalidResourceHandle);
  }

  SECTION("With invalid attribute") {
    attr = static_cast<hipStreamAttrID>(-1);
    HIP_CHECK_ERROR(hipStreamSetAttribute(stream, attr, &valueToSet),
                    hipErrorInvalidValue);
  }
#if HT_AMD
  // Facing segmentation fault issue in CUDA
  SECTION("With invalid attribute value") {
    HIP_CHECK_ERROR(hipStreamSetAttribute(stream, attr, nullptr),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("With Invalid hipSynchronizationPolicy") {
    valueToSet.syncPolicy = static_cast<hipSynchronizationPolicy>(-1);
    HIP_CHECK_ERROR(hipStreamSetAttribute(stream, attr, &valueToSet),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipStreamDestroy(stream));
}
