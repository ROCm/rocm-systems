/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipStreamWaitEvent_Negative - Test unsuccessful hipStreamWaitEvent when either event or flags
are invalid Unit_hipStreamWaitEvent_Default - Test simple waiting for an event with
hipStreamWaitEvent api Unit_hipStreamWaitEvent_DifferentStreams - Test waiting for an event on a
different stream with hipStreamWaitEvent api
Unit_hipStreamWaitEvent_PendingEventEmptyWaiter - A stream whose only work is a wait on an event
that is still pending on another stream must complete once that event completes
*/

#include <hip_test_common.hh>
#include <utils.hh>

#include <chrono>
#include <thread>
HIP_TEST_CASE(Unit_hipStreamWaitEvent_Negative) {
  enum class StreamTestType { NullStream = 0, StreamPerThread, CreatedStream };

  auto streamType = GENERATE(StreamTestType::NullStream, StreamTestType::StreamPerThread,
                             StreamTestType::CreatedStream);

  hipStream_t stream{nullptr};
  hipEvent_t event{nullptr};

  if (streamType == StreamTestType::StreamPerThread) {
    stream = hipStreamPerThread;
  } else if (streamType == StreamTestType::CreatedStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  HIP_CHECK(hipEventCreate(&event));

  REQUIRE((stream != nullptr) != (streamType == StreamTestType::NullStream));
  REQUIRE(event != nullptr);

  SECTION("Invalid Event") {
    INFO("Running against Invalid Event");
    HIP_CHECK_ERROR(hipStreamWaitEvent(stream, nullptr, 0), hipErrorInvalidResourceHandle);
  }

  SECTION("Invalid Flags") {
    INFO("Running against Invalid Flags");
    constexpr unsigned flag = ~0u;
    REQUIRE(flag != 0);
    HIP_CHECK_ERROR(hipStreamWaitEvent(stream, event, flag), hipErrorInvalidValue);
  }

  HIP_CHECK(hipEventDestroy(event));

  if (streamType == StreamTestType::CreatedStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

HIP_TEST_CASE(Unit_hipStreamWaitEvent_Default) {
  hipStream_t stream{nullptr};
  hipEvent_t waitEvent{nullptr};

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreate(&waitEvent));

  REQUIRE(stream != nullptr);
  REQUIRE(waitEvent != nullptr);

  LaunchDelayKernel(std::chrono::milliseconds(isQuickLevel() ? 100 : 2000), stream);

  HIP_CHECK(hipEventRecord(waitEvent, stream));

  // Make sure stream is waiting for data to be set
  HIP_CHECK_ERROR(hipEventQuery(waitEvent), hipErrorNotReady);

  HIP_CHECK(hipStreamWaitEvent(stream, waitEvent, 0));

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(waitEvent));
}

HIP_TEST_CASE(Unit_hipStreamWaitEvent_DifferentStreams) {
  hipStream_t blockedStreamA{nullptr}, streamBlockedOnStreamA{nullptr}, unblockingStream{nullptr};
  hipEvent_t waitEvent{nullptr};

  HIP_CHECK(hipStreamCreate(&blockedStreamA));
  HIP_CHECK(hipStreamCreate(&streamBlockedOnStreamA));
  HIP_CHECK(hipStreamCreate(&unblockingStream));
  HIP_CHECK(hipEventCreate(&waitEvent));

  REQUIRE(blockedStreamA != nullptr);
  REQUIRE(streamBlockedOnStreamA != nullptr);
  REQUIRE(waitEvent != nullptr);

  LaunchDelayKernel(std::chrono::milliseconds(3000), blockedStreamA);

  HIP_CHECK(hipEventRecord(waitEvent, blockedStreamA));

  // Make sure stream is waiting for data to be set
  HIP_CHECK_ERROR(hipEventQuery(waitEvent), hipErrorNotReady);

  HIP_CHECK(hipStreamWaitEvent(streamBlockedOnStreamA, waitEvent, 0));

  LaunchDelayKernel(std::chrono::milliseconds(2000), streamBlockedOnStreamA);

  HIP_CHECK(hipStreamSynchronize(unblockingStream));

  HIP_CHECK(hipStreamSynchronize(blockedStreamA));

  // Make sure streamBlockedOnStreamA waited for event on blockedStreamA
  HIP_CHECK_ERROR(hipStreamQuery(streamBlockedOnStreamA), hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(streamBlockedOnStreamA));

  // Check that both streams have finished
  HIP_CHECK(hipStreamQuery(blockedStreamA));
  HIP_CHECK(hipStreamQuery(streamBlockedOnStreamA));

  HIP_CHECK(hipStreamDestroy(blockedStreamA));
  HIP_CHECK(hipStreamDestroy(streamBlockedOnStreamA));
  HIP_CHECK(hipStreamDestroy(unblockingStream));
  HIP_CHECK(hipEventDestroy(waitEvent));
}

namespace {
__global__ void IncrementCounter(unsigned* counter) { ++*counter; }

// Polls instead of calling hipStreamSynchronize, so a stream that never completes fails the
// test instead of blocking it.
bool StreamCompletesWithin(hipStream_t stream, std::chrono::seconds limit) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  while (std::chrono::steady_clock::now() < deadline) {
    const hipError_t status = hipStreamQuery(stream);
    if (status == hipSuccess) return true;
    HIP_CHECK_ERROR(status, hipErrorNotReady);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - A producer stream runs many small kernels, as a graph launch or as eager launches,
 *    and records an event. The event is still pending when a second stream waits on it.
 *    The waiting stream has no other work.
 *  - After the event completes, the waiting stream must also complete.
 *  - Repeats the sequence so that the runtime reuses its completion signals.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamWaitEvent.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamWaitEvent_PendingEventEmptyWaiter) {
  constexpr int kKernels = 2048;
  constexpr int kTrials = 20;
  const bool useGraph = GENERATE(true, false);
  INFO("producer: " << (useGraph ? "graph launch" : "eager launches"));

  hipStream_t producer{nullptr}, waiter{nullptr};
  hipEvent_t done{nullptr};
  unsigned* counter{nullptr};
  HIP_CHECK(hipStreamCreateWithFlags(&producer, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&waiter, hipStreamNonBlocking));
  HIP_CHECK(hipEventCreate(&done));
  HIP_CHECK(hipMalloc(&counter, sizeof(unsigned)));

  hipGraphExec_t exec{nullptr};
  if (useGraph) {
    hipGraph_t graph{nullptr};
    HIP_CHECK(hipStreamBeginCapture(producer, hipStreamCaptureModeGlobal));
    for (int i = 0; i < kKernels; ++i) {
      hipLaunchKernelGGL(IncrementCounter, dim3(1), dim3(1), 0, producer, counter);
    }
    HIP_CHECK(hipStreamEndCapture(producer, &graph));
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  int pendingAtWait = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    INFO("trial " << trial);
    HIP_CHECK(hipMemsetAsync(counter, 0, sizeof(unsigned), producer));
    HIP_CHECK(hipStreamSynchronize(producer));
    if (useGraph) {
      HIP_CHECK(hipGraphLaunch(exec, producer));
    } else {
      for (int i = 0; i < kKernels; ++i) {
        hipLaunchKernelGGL(IncrementCounter, dim3(1), dim3(1), 0, producer, counter);
      }
    }
    HIP_CHECK(hipEventRecord(done, producer));
    const hipError_t query = hipEventQuery(done);
    if (query != hipErrorNotReady) HIP_CHECK(query);
    pendingAtWait += (query == hipErrorNotReady);
    HIP_CHECK(hipStreamWaitEvent(waiter, done, 0));
    HIP_CHECK(hipEventSynchronize(done));
    REQUIRE(StreamCompletesWithin(waiter, std::chrono::seconds(10)));
    unsigned result = 0;
    HIP_CHECK(hipMemcpy(&result, counter, sizeof(result), hipMemcpyDeviceToHost));
    REQUIRE(result == static_cast<unsigned>(kKernels));
  }
  // The scenario needs the event to be pending at the wait in at least one trial.
  REQUIRE(pendingAtWait > 0);

  if (exec != nullptr) HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipFree(counter));
  HIP_CHECK(hipEventDestroy(done));
  HIP_CHECK(hipStreamDestroy(waiter));
  HIP_CHECK(hipStreamDestroy(producer));
}
