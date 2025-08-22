/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
/*
Testcase Scenarios :
Unit_hipStreamWaitEvent_Negative - Test unsuccessful hipStreamWaitEvent when either event or flags
are invalid Unit_hipStreamWaitEvent_Default - Test simple waiting for an event with
hipStreamWaitEvent api Unit_hipStreamWaitEvent_DifferentStreams - Test waiting for an event on a
different stream with hipStreamWaitEvent api
*/

#include <hip_test_common.hh>
#include <utils.hh>
TEST_CASE("Unit_hipStreamWaitEvent_Negative") {
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

TEST_CASE("Unit_hipStreamWaitEvent_Default") {
  hipStream_t stream{nullptr};
  hipEvent_t waitEvent{nullptr};

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreate(&waitEvent));

  REQUIRE(stream != nullptr);
  REQUIRE(waitEvent != nullptr);

  LaunchDelayKernel(std::chrono::milliseconds(2000), stream);

  HIP_CHECK(hipEventRecord(waitEvent, stream));

  // Make sure stream is waiting for data to be set
  HIP_CHECK_ERROR(hipEventQuery(waitEvent), hipErrorNotReady);

  HIP_CHECK(hipStreamWaitEvent(stream, waitEvent, 0));

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(waitEvent));
}

TEST_CASE("Unit_hipStreamWaitEvent_DifferentStreams") {
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
