/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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

HIP_TEST_CASE(Unit_hipStreamCopyAttributes_Basic) {
  hipStream_t stream1, stream2, stream3, stream4;
  hipStreamAttrValue val1, val2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  SECTION("Two Non Default Streams") {
    val1.syncPolicy = hipSyncPolicySpin;

    // Set the sync policy attribute of stream1 to hipSyncPolicySpin
    HIP_CHECK(hipStreamSetAttribute(
        stream1, hipStreamAttributeSynchronizationPolicy, &val1));

    // Copy attributes from stream1 to stream2
    HIP_CHECK(hipStreamCopyAttributes(stream2, stream1));

    // Query stream2 to verify the copied sync policy
    HIP_CHECK(hipStreamGetAttribute(
        stream2, hipStreamAttributeSynchronizationPolicy, &val2));

    REQUIRE(val2.syncPolicy == hipSyncPolicySpin);
  }

  SECTION("Copy attributes from Null Stream to Legacy Stream") {
    stream3 = nullptr;
    stream4 = hipStreamLegacy;
    val1.syncPolicy = hipSyncPolicyYield;

    // Set the sync policy attribute of stream1 to hipSyncPolicySpin
    HIP_CHECK(hipStreamSetAttribute(
        stream3, hipStreamAttributeSynchronizationPolicy, &val1));

    // Copy attributes from null stream to legacy stream
    HIP_CHECK(hipStreamCopyAttributes(stream4, stream3));

    // Query stream2 to verify the copied sync policy
    HIP_CHECK(hipStreamGetAttribute(
        stream4, hipStreamAttributeSynchronizationPolicy, &val2));

    REQUIRE(val2.syncPolicy == hipSyncPolicyYield);
  }

  SECTION("Copy attributes from streamperthread to another stream") {
    stream3 = hipStreamPerThread;
    val1.syncPolicy = hipSyncPolicyBlockingSync;

    // Set the sync policy attribute of stream1 to hipSyncPolicySpin
    HIP_CHECK(hipStreamSetAttribute(
        stream3, hipStreamAttributeSynchronizationPolicy, &val1));

    // Copy attributes from streamperthread to non default stream
    HIP_CHECK(hipStreamCopyAttributes(stream2, stream3));

    // Query stream2 to verify the copied sync policy
    HIP_CHECK(hipStreamGetAttribute(
        stream2, hipStreamAttributeSynchronizationPolicy, &val2));

    REQUIRE(val2.syncPolicy == hipSyncPolicyBlockingSync);
  }

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following negative scenarios
 *  - 1) With Invalid source Stream
 *  - 2) With Invalid destination Stream
 *  - 3) With Invalid source and destination Streams
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCopyAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamCopyAttributes_Negative) {
  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));
  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));

  hipStream_t invalidStream = reinterpret_cast<hipStream_t>(-1);

  SECTION("Sanity - Should pass") {
    HIP_CHECK(hipStreamCopyAttributes(srcStream, dstStream));
  }

  SECTION("With Invalid Source Stream") {
    HIP_CHECK_ERROR(hipStreamCopyAttributes(dstStream, invalidStream),
                    hipErrorInvalidResourceHandle);
  }

  SECTION("With Invalid Destination Stream") {
    HIP_CHECK_ERROR(hipStreamCopyAttributes(invalidStream, srcStream),
                    hipErrorInvalidResourceHandle);
  }

  SECTION("With Invalid Source & Destination Streams") {
    HIP_CHECK_ERROR(hipStreamCopyAttributes(invalidStream, invalidStream),
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
HIP_TEST_CASE(Unit_hipStreamCopyAttributes_WithAllSyncPolicyValues) {
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
