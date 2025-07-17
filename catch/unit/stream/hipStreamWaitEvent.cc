/*
Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../memory/mempool_common.hh"
#include <utils.hh>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>

using namespace std;

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
#if HT_NVIDIA
TEST_CASE("Unit_hipStreamWaitEvent_StreamCapture_WithCross_Dependency") {
  GENERATE_CAPTURE();
  streamMemAllocTest testObj(N);
  // create multiple streams
  hipStream_t stream1, stream2, stream3;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  // Create host buffer with test data
  testObj.createHostBufferWithData();
  hipEvent_t Event1, Event2, Event3, Event4;
  HIP_CHECK(hipEventCreate(&Event1));
  HIP_CHECK(hipEventCreate(&Event2));
  HIP_CHECK(hipEventCreate(&Event3));
  HIP_CHECK(hipEventCreate(&Event4));
  BEGIN_CAPTURE(stream1);
  testObj.allocFromDefMempool(stream1);
  testObj.transferToMempool(stream1);
  HIP_CHECK(hipEventRecord(Event1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, Event1, 0));
  testObj.runKernel(stream2);
  HIP_CHECK(hipEventRecord(Event2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream3, Event2, 0));
  testObj.transferFromMempool(stream3);
  testObj.freeDevBuf(stream3);
  HIP_CHECK(hipEventRecord(Event3, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream2, Event3, 0));
  HIP_CHECK(hipEventRecord(Event4, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, Event4, 0));
  END_CAPTURE(stream1);
  HIP_CHECK(hipStreamSynchronize(stream1));
  // Validate the Result;
  REQUIRE(true == testObj.validateResult());
  testObj.freeHostBuf();
  HIP_CHECK(hipEventDestroy(Event4));
  HIP_CHECK(hipEventDestroy(Event3));
  HIP_CHECK(hipEventDestroy(Event2));
  HIP_CHECK(hipEventDestroy(Event1));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
condition_variable cvar;
mutex mut;
int var = 0;
void threadFunc_3(hipStream_t stream3, streamMemAllocTest testObj) {
  hipEvent_t Event3, Event4;
  hipStream_t stream4;
  HIP_CHECK(hipEventCreate(&Event3));
  HIP_CHECK(hipEventCreate(&Event4));
  HIP_CHECK(hipStreamCreate(&stream4));
  HIP_CHECK(hipEventRecord(Event3, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream4, Event3, 0));
  testObj.transferFromMempool(stream4);
  testObj.freeDevBuf(stream4);
  HIP_CHECK(hipEventRecord(Event4, stream4));
  HIP_CHECK(hipStreamWaitEvent(stream3, Event4, 0));
}

void threadFunc_2(hipStream_t stream1, streamMemAllocTest testObj) {
  hipEvent_t Event2, Event5;
  hipStream_t stream3;
  HIP_CHECK(hipEventCreate(&Event2));
  HIP_CHECK(hipEventCreate(&Event5));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipEventRecord(Event2, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream3, Event2, 0));
  unique_lock<mutex> ulock(mut);
  cvar.wait(ulock, [] { return (var == 1) ? true : false; });
  testObj.runKernel(stream3);
  std::thread t3(threadFunc_3, stream3, testObj);
  t3.join();
  HIP_CHECK(hipEventRecord(Event5, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, Event5, 0));
}

void threadFunc_1(hipStream_t stream1, streamMemAllocTest testObj) {
  lock_guard<mutex> lock(mut);
  hipEvent_t Event1, Event6;
  hipStream_t stream2;
  HIP_CHECK(hipEventCreate(&Event1));
  HIP_CHECK(hipEventCreate(&Event6));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventRecord(Event1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, Event1, 0));
  testObj.transferToMempool(stream2);
  var = 1;
  cvar.notify_one();
  HIP_CHECK(hipEventRecord(Event6, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, Event6, 0));
}

TEST_CASE("Unit_hipStreamWaitEvent_StreamCapture_CrossDepend_MultiThread") {
  hipStream_t stream1;
  HIP_CHECK(hipStreamCreate(&stream1));
  streamMemAllocTest testObj(512);
  testObj.createHostBufferWithData();
  hipStreamCaptureMode flags = GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream1, flags));
  testObj.allocFromDefMempool(stream1);
  std::thread t1(threadFunc_1, stream1, testObj);
  std::thread t2(threadFunc_2, stream1, testObj);
  t1.join();
  t2.join();
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  // Validate the Result;
  REQUIRE(true == testObj.validateResult());
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  testObj.freeHostBuf();
}
#endif
