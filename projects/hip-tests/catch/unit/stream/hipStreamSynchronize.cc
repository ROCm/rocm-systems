/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "streamCommon.hh"
#include <utils.hh>
namespace hipStreamSynchronizeTest {

/**
 * @brief Check that hipStreamSynchronize handles empty streams properly.
 *
 */
HIP_TEST_CASE(Unit_hipStreamSynchronize_EmptyStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */

/**
 * @brief Check that all work executing in a stream is finished after a call to
 * hipStreamSynchronize.
 *
 */
HIP_TEST_CASE(Unit_hipStreamSynchronize_FinishWork) {
  const hipStream_t explicitStream = reinterpret_cast<hipStream_t>(-1);
  hipStream_t stream = GENERATE_COPY(explicitStream, hip::nullStream, hip::streamPerThread);

  const bool isExplicitStream = stream == explicitStream;
  if (isExplicitStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  LaunchDelayKernel(std::chrono::milliseconds(500), stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamQuery(stream));

  if (isExplicitStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * @brief Check that synchronizing the nullStream implicitly synchronizes all executing streams.
 */
HIP_TEST_CASE(Unit_hipStreamSynchronize_NullStreamSynchronization) {
  int totalStreams = 10;

  std::vector<hipStream_t> streams{};

  for (int i = 0; i < totalStreams; ++i) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    streams.push_back(stream);
  }

  for (int i = 0; i < totalStreams; ++i) {
    LaunchDelayKernel(std::chrono::milliseconds(1000), streams[i]);
  }

  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK_ERROR(hipStreamQuery(streams[i]), hipErrorNotReady);
  }

  HIP_CHECK(hipStreamSynchronize(hip::nullStream));
  HIP_CHECK(hipStreamQuery(hip::nullStream));

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamQuery(streams[i]));
  }

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

/**
 * @brief Check that synchronizing one stream does implicitly synchronize other streams.
 *        Check that submiting work to the nullStream does not affect synchronization of other
 * streams. Check that querying the nullStream does not affect synchronization of other streams.
 */
HIP_TEST_CASE(Unit_hipStreamSynchronize_SynchronizeStreamAndQueryNullStream) {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-22");
#else

  hipStream_t stream1;
  hipStream_t stream2;

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  LaunchDelayKernel(std::chrono::milliseconds(500), stream1);
  LaunchDelayKernel(std::chrono::milliseconds(2000), stream2);

  SECTION("Do not use NullStream") {}
  SECTION("Submit Kernel to NullStream") {
    hip::stream::empty_kernel<<<1, 1, 0, hip::nullStream> > >();
  }
  SECTION("Query NullStream") {
    HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
  }

  HIP_CHECK_ERROR(hipStreamQuery(stream1), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);


  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamQuery(stream1));
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize(stream2));
  HIP_CHECK(hipStreamQuery(stream2));

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
#endif
}

/**
 * @brief Check that synchronizing the nullStream also synchronizes the hipStreamPerThread
 * special stream.
 *
 */
HIP_TEST_CASE(Unit_hipStreamSynchronize_NullStreamAndStreamPerThread) {
  LaunchDelayKernel(std::chrono::milliseconds(500), hip::streamPerThread);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::streamPerThread), hipErrorNotReady);
  LaunchDelayKernel(std::chrono::milliseconds(500), hip::nullStream);
  HIP_CHECK(hipStreamSynchronize(hip::nullStream))
  HIP_CHECK_ERROR(hipStreamQuery(hip::streamPerThread), hipSuccess);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipSuccess);
}
#endif
}  // namespace hipStreamSynchronizeTest
