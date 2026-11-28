/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "streamCommon.hh"
#include <utils.hh>
#include <resource_guards.hh>

#include <future>
#include <thread>
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

  LaunchDelayKernel(std::chrono::milliseconds(isQuickLevel() ? 50 : 500), stream);
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
    LaunchDelayKernel(std::chrono::milliseconds(isQuickLevel() ? 100 : 1000), streams[i]);
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
 * @brief Check that synchronizing an already completed stream does not wait for
 * unrelated pending work, including after a null-stream launch or query.
 */
HIP_TEST_CASE(Unit_hipStreamSynchronize_SynchronizeStreamAndQueryNullStream) {
  const StreamGuard stream1(Streams::created);
  const StreamGuard stream2(Streams::created);
  LinearAllocGuard<int> release(LinearAllocs::hipHostMalloc, sizeof(int));
  *release.ptr() = 0;

  hip::stream::empty_kernel<<<1, 1, 0, stream1.stream()>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream1.stream()));

  // Complete stream1 before holding stream2 so shared hardware queues cannot
  // prevent stream1 from progressing. Do not enqueue new stream1 work after
  // touching the null stream: such work would correctly wait for stream2.
  WaitForHostRelease<<<1, 1, 0, stream2.stream()>>>(release.ptr());
  const auto launchError = hipGetLastError();
  std::promise<void> queriesDone;
  std::thread releaser([flag = release.ptr(), done = queriesDone.get_future()] {
    // Bound a regression that incorrectly waits for stream2 on the host.
    done.wait_for(std::chrono::seconds(10));
    __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
  });

  hipError_t nullActionError = hipSuccess;
  hipError_t expectedNullActionError = hipSuccess;
  SECTION("Do not use NullStream") {}
  SECTION("Submit Kernel to NullStream") {
    hip::stream::empty_kernel<<<1, 1, 0, hip::nullStream>>>();
    nullActionError = hipGetLastError();
  }
  SECTION("Query NullStream") {
    nullActionError = hipStreamQuery(hip::nullStream);
    expectedNullActionError = hipErrorNotReady;
  }

  // Capture statuses before releasing stream2, and join before Catch assertions
  // so a failing check cannot strand a GPU kernel or a joinable host thread.
  const auto sync1Error = hipStreamSynchronize(stream1.stream());
  const auto query1Error = hipStreamQuery(stream1.stream());
  const auto query2Error = hipStreamQuery(stream2.stream());
  const auto queryNullError = hipStreamQuery(hip::nullStream);
  queriesDone.set_value();
  releaser.join();
  HIP_CHECK(hipStreamSynchronize(stream2.stream()));
  HIP_CHECK(hipStreamSynchronize(hip::nullStream));

  HIP_CHECK(launchError);
  HIP_CHECK_ERROR(nullActionError, expectedNullActionError);
  HIP_CHECK(sync1Error);
  HIP_CHECK(query1Error);
  HIP_CHECK_ERROR(query2Error, hipErrorNotReady);
  HIP_CHECK_ERROR(queryNullError, hipErrorNotReady);
  HIP_CHECK(hipStreamQuery(stream2.stream()));
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
