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

/**
 * @addtogroup hipStreamCopyAttributes hipStreamCopyAttributes
 * @{
 * @ingroup StreamTest
 * `hipStreamCopyAttributes (hipStream_t dst, hipStream_t src)` -
 * copies attributes from one stream to other 
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *    - Test that creates two streams and copies attributes from one to another
 * ------------------------
 *    - catch\unit\stream\hipStreamCopyAttributes.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */

TEST_CASE("Unit_hipStreamCopyAttributes_Basic") {
  hipStream_t stream1, stream2, stream3, stream4;
  hipStreamAttrValue val1, val2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
 
  SECTION("Two Non Default Streams") {
    val1.syncPolicy = hipSyncPolicySpin;

    // Set the sync policy attribute of stream1 to hipSyncPolicySpin
    HIP_CHECK(hipStreamSetAttribute(stream1, hipStreamAttributeSynchronizationPolicy, &val1));

    // Copy attributes from stream1 to stream2
    HIP_CHECK(hipStreamCopyAttributes(stream2, stream1));

    // Query stream2 to verify the copied sync policy
    HIP_CHECK(hipStreamGetAttribute(stream2, hipStreamAttributeSynchronizationPolicy, &val2));

    REQUIRE(val2.syncPolicy == hipSyncPolicySpin);
  }

  SECTION("Copy attributes from Null Stream to Legacy Stream") {
    stream3 = nullptr;
    stream4 = hipStreamLegacy;
    val1.syncPolicy = hipSyncPolicyYield;

    // Set the sync policy attribute of stream1 to hipSyncPolicySpin
    HIP_CHECK(hipStreamSetAttribute(stream3, hipStreamAttributeSynchronizationPolicy, &val1));

    // Copy attributes from null stream to legacy stream
    HIP_CHECK(hipStreamCopyAttributes(stream4, stream3));

    // Query stream2 to verify the copied sync policy
    HIP_CHECK(hipStreamGetAttribute(stream4, hipStreamAttributeSynchronizationPolicy, &val2));

    REQUIRE(val2.syncPolicy == hipSyncPolicyYield);
  }

  SECTION("Copy attributes from streamperthread to another stream") {
    stream3 = hipStreamPerThread;
    val1.syncPolicy = hipSyncPolicyBlockingSync;

    // Set the sync policy attribute of stream1 to hipSyncPolicySpin
    HIP_CHECK(hipStreamSetAttribute(stream3, hipStreamAttributeSynchronizationPolicy, &val1));

    // Copy attributes from streamperthread to non default stream
    HIP_CHECK(hipStreamCopyAttributes(stream2, stream3));

    // Query stream2 to verify the copied sync policy
    HIP_CHECK(hipStreamGetAttribute(stream2, hipStreamAttributeSynchronizationPolicy, &val2));

    REQUIRE(val2.syncPolicy == hipSyncPolicyBlockingSync);
  }
 
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}
