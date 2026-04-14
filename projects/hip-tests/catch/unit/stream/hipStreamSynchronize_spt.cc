/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <utils.hh>
#include "streamCommon.hh"
/**
 * @addtogroup hipStreamSynchronize_spt hipStreamSynchronize_spt
 * @{
 * @ingroup StreamTest
 * `hipStreamSynchronize_spt(hipStream_t stream)` -
 * Query the priority of a stream
 */

/**
 * Test Description
 * ------------------------
 * - Test to verify that hipStreamSynchronize_spt handles empty streams properly.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamSynchronize_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipStreamSynchronize_spt_EmptyStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamSynchronize_spt(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Check that synchronization of uninitialized stream sets its status to
 * hipErrorContextIsDestroyed
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamSynchronize_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

#if HT_AMD
/**
 * Test Description
 * ------------------------
 * - Check that all work executing in a stream is finished after a call to
 * hipStreamSynchronize_spt.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamSynchronize_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipStreamSynchronize_spt_FinishWork) {
  hipStream_t explicitStream = reinterpret_cast<hipStream_t>(-1);
  hipStream_t stream = GENERATE_COPY(explicitStream, hip::nullStream, hip::streamPerThread);
  if (explicitStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  LaunchDelayKernel(std::chrono::milliseconds(500), stream);
  HIP_CHECK(hipStreamSynchronize_spt(stream));
  HIP_CHECK(hipStreamQuery(stream));

  if (explicitStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}
/**
 * Test Description
 * ------------------------
 * - Check that synchronizing one stream does implicitly synchronize
 * other streams. Check that submiting work to the nullStream does
 * not affect synchronization of other streams. Check that querying
 * the nullStream does not affect synchronization of other streams.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamSynchronize_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

TEST_CASE(Unit_hipStreamSynchronize_spt_SynchronizeStreamAndQueryNullStream) {
  hipStream_t stream1;
  hipStream_t stream2;

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  LaunchDelayKernel(std::chrono::milliseconds(500), stream1);
  LaunchDelayKernel(std::chrono::milliseconds(2000), stream2);
  hip::stream::empty_kernel<<<1, 1, 0, hip::nullStream>>>();

  HIP_CHECK_ERROR(hipStreamQuery(stream1), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize_spt(stream1));
  HIP_CHECK(hipStreamQuery(stream1));
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize_spt(stream2));
  HIP_CHECK(hipStreamQuery(stream2));

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}
#endif
/**
 * End doxygen group StreamTest.
 * @}
 */
