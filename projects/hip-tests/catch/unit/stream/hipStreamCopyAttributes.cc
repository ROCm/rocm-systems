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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following negative scenarios
 *  - 1) With Invalid source Stream
 *  - 2) With Invalid destination Stream
 *  - 3) With Invalid source and destination Stream
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCopyAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipStreamCopyAttributes_Negative") {
  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));
  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));

  SECTION("With Invalid Source Stream") {
    HIP_CHECK_ERROR(
        hipStreamCopyAttributes(dstStream, reinterpret_cast<hipStream_t>(-1)),
        hipErrorInvalidResourceHandle);
  }

  SECTION("With Invalid Destination Stream") {
    HIP_CHECK_ERROR(
        hipStreamCopyAttributes(reinterpret_cast<hipStream_t>(-1), srcStream),
        hipErrorInvalidResourceHandle);
  }

  SECTION("With Invalid Source & Destination Streams") {
    HIP_CHECK_ERROR(hipStreamCopyAttributes(reinterpret_cast<hipStream_t>(-1),
                                            reinterpret_cast<hipStream_t>(-1)),
                    hipErrorInvalidResourceHandle);
  }

  HIP_CHECK(hipStreamDestroy(srcStream));
  HIP_CHECK(hipStreamDestroy(dstStream));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks behavior of hipStreamCopyAttributes
 *  - with SynchronizationPolicy attribute and with all possible values
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCopyAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipStreamCopyAttributes_WithAllSyncPolicyValues") {
  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));
  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueToSetForSrc;
  hipSynchronizationPolicy syncPolicy =
      GENERATE(hipSynchronizationPolicy::hipSyncPolicyAuto,
               hipSynchronizationPolicy::hipSyncPolicySpin,
               hipSynchronizationPolicy::hipSyncPolicyYield,
               hipSynchronizationPolicy::hipSyncPolicyBlockingSync);
  valueToSetForSrc.syncPolicy = syncPolicy;
  HIP_CHECK(hipStreamSetAttribute(srcStream, attr, &valueToSetForSrc));

  hipStreamAttrValue valueToSetForDst;
  valueToSetForDst.syncPolicy = hipSynchronizationPolicy::hipSyncPolicySpin;
  HIP_CHECK(hipStreamSetAttribute(dstStream, attr, &valueToSetForDst));

  HIP_CHECK(hipStreamCopyAttributes(dstStream, srcStream));

  hipStreamAttrValue valueOut;
  HIP_CHECK(hipStreamGetAttribute(dstStream, attr, &valueOut));
  REQUIRE(valueOut.syncPolicy == syncPolicy);

  HIP_CHECK(hipStreamDestroy(srcStream));
  HIP_CHECK(hipStreamDestroy(dstStream));
}
