/*
Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../memory/mempool_common.hh"
#include "streamCommon.hh"
#include <condition_variable>
#include <mutex>
#include <thread>
using namespace std;
static constexpr auto NUM_ELM{1024 * 1024};
size_t byte_size = NUM_ELM * sizeof(int);
/**
 * @addtogroup hipStreamCreate hipStreamCreate
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamCreate(hipStream_t* stream);` -
 * Creates an asynchronous stream.
 */

/**
 * Test Description
 * ------------------------
 * - Test to verifies the basic functionality of hipStreamCreate.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamCreate_default") {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);        // Check if stream has a valid ptr
  REQUIRE(hip::checkStream(stream)); // check its flags and priority
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Test to verifies the negative case of hipStreamCreate.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_hipStreamCreate_Negative") {
  REQUIRE(hipErrorInvalidValue == hipStreamCreate(nullptr));
}
/**
 * Test Description
 * ------------------------
 * - Test to verifies the following case.
 * 1. Enable capture on stream s1.
 * 2. Enable capture again on s1, returns hipErrorIllegalState.
 * 3. End capture on s1.
 * Capture can be initiated if the stream is not already in capture mode
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_hipStreamCreate_StreamCapture_NegTst") {
  HIP_CHECK(hipSetDevice(0));

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);
  hipStreamCaptureMode flag =
      GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
               hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream, flag));
  HIP_CHECK_ERROR(hipStreamBeginCapture(stream, flag), hipErrorIllegalState);
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph))
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Test to verifies the following case.
 * - capture different graphs on multiple stream parallely from same thread.
 * 1. Enable capture on stream s1, s2
 * 2. Enqueue work on capture stream s1, s2
 * 3. End capture on s1, s2.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamCreate_StreamCapture_ParallelCapture") {
  // create multiple streams
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  streamMemAllocTest testObj1(byte_size);
  streamMemAllocTest testObj2(byte_size);
  testObj1.createHostBufferWithData();
  testObj2.createHostBufferWithData();
  hipStreamCaptureMode flags =
      GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
               hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream1, flags));
  HIP_CHECK(hipStreamBeginCapture(stream2, flags));
  testObj1.allocFromDefMempool(stream1);
  testObj2.allocFromDefMempool(stream2);
  testObj1.transferToMempool(stream1);
  testObj2.transferToMempool(stream2);
  testObj1.runKernel(stream1);
  testObj2.runKernel(stream2);
  testObj1.transferFromMempool(stream1);
  testObj2.transferFromMempool(stream2);
  testObj1.freeDevBuf(stream1);
  testObj2.freeDevBuf(stream2);
  hipGraph_t graph1 = nullptr, graph2 = nullptr;
  hipGraphExec_t graph_exec1 = nullptr, graph_exec2 = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
  HIP_CHECK(hipGraphInstantiate(&graph_exec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec1, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  REQUIRE(true == testObj1.validateResult());
  HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  HIP_CHECK(hipGraphDestroy(graph1));

  HIP_CHECK(hipGraphInstantiate(&graph_exec2, graph2, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec2, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));
  REQUIRE(true == testObj2.validateResult());
  HIP_CHECK(hipGraphExecDestroy(graph_exec2));
  HIP_CHECK(hipGraphDestroy(graph2));
  // Destroy resources
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  testObj1.freeHostBuf();
  testObj2.freeHostBuf();
}

void threadFunc_1(hipStream_t stream1, size_t byte_size,
                  hipStreamCaptureMode flags) {
  streamMemAllocTest testObj(byte_size);
  testObj.createHostBufferWithData();
  HIP_CHECK(hipStreamBeginCapture(stream1, flags));
  testObj.allocFromDefMempool(stream1);
  testObj.transferToMempool(stream1);
  testObj.runKernel(stream1);
  testObj.transferFromMempool(stream1);
  testObj.freeDevBuf(stream1);
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  REQUIRE(true == testObj.validateResult());
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  testObj.freeHostBuf();
}

void threadFunc_2(hipStream_t stream2, size_t byte_size,
                  hipStreamCaptureMode flags) {
  streamMemAllocTest testObj(byte_size);
  testObj.createHostBufferWithData();
  HIP_CHECK(hipStreamBeginCapture(stream2, flags));
  testObj.allocFromDefMempool(stream2);
  testObj.transferToMempool(stream2);
  testObj.runKernel(stream2);
  testObj.transferFromMempool(stream2);
  testObj.freeDevBuf(stream2);
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream2, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));
  REQUIRE(true == testObj.validateResult());
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  testObj.freeHostBuf();
}
/**
 * Test Description
 * ------------------------
 * - Test to verifies the following case.
 * - capture different graphs on multiple streams parallely from different
 * threads
 * 1. Enable capture on stream s1 from thread1
 * 2. Enable capture on stream s2 from thread2
 * 3. Enqueue work on capture stream s1, s2
 * 4. End capture on s1, s2 from respective threads
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamCreate_StreamCapture_ParallelCapture_MultiThrd") {
  // create multiple streams
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  hipStreamCaptureMode flags =
      GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
               hipStreamCaptureModeRelaxed);
  thread t1(threadFunc_1, stream1, byte_size, flags);
  thread t2(threadFunc_2, stream2, byte_size, flags);
  t1.join();
  t2.join();
  // Destroy streams
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}

condition_variable cv;
mutex m;
streamMemAllocTest testObj1(byte_size);
streamMemAllocTest testObj2(byte_size);
int val = 0;
void threadFuncDiff_1(hipStream_t stream1, hipStream_t stream2) {
  testObj1.createHostBufferWithData();
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeRelaxed));
  testObj1.allocFromDefMempool(stream1);
  testObj1.transferToMempool(stream1);
  testObj1.runKernel(stream1);
  testObj1.transferFromMempool(stream1);
  testObj1.freeDevBuf(stream1);
  hipGraph_t graph1 = nullptr;
  hipGraphExec_t graph_exec1 = nullptr;
  unique_lock<mutex> ulock(m);
  cv.wait(ulock, [] { return (val == 1) ? true : false; });
  HIP_CHECK(hipStreamEndCapture(stream2, &graph1));
  HIP_CHECK(hipGraphInstantiate(&graph_exec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec1, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));
  REQUIRE(true == testObj2.validateResult());
  HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  HIP_CHECK(hipGraphDestroy(graph1));
  testObj2.freeHostBuf();
}

void threadFuncDiff_2(hipStream_t stream1, hipStream_t stream2) {
  lock_guard<mutex> lock(m);
  testObj2.createHostBufferWithData();
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeRelaxed));
  testObj2.allocFromDefMempool(stream2);
  testObj2.transferToMempool(stream2);
  testObj2.runKernel(stream2);
  testObj2.transferFromMempool(stream2);
  testObj2.freeDevBuf(stream2);
  this_thread::sleep_for(chrono::seconds(1));
  val = 1;
  cv.notify_one();
  hipGraph_t graph2 = nullptr;
  hipGraphExec_t graph_exec2 = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream1, &graph2));
  HIP_CHECK(hipGraphInstantiate(&graph_exec2, graph2, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec2, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  REQUIRE(true == testObj1.validateResult());
  HIP_CHECK(hipGraphExecDestroy(graph_exec2));
  HIP_CHECK(hipGraphDestroy(graph2));
  testObj1.freeHostBuf();
}
/**
 * Test Description
 * ------------------------
 * - Test to verifies the following case.
 * - capture different graphs on multiple streams parallely from different
 * threads.
 * - (If mode is not hipStreamCaptureModeRelaxed, hipStreamEndCapture must be
 * called
 *  - on this stream from the same thread.)
 * 1. Enable capture on stream s1 from thread1
 * 2. Enable capture on stream s2 from thread2
 * 3. Enqueue work on capture stream s1, s2
 * 4. End capture on s1, s2 from different threads
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamCreate_StreamCapture_DiffCapture_MultiThrd") {
  // create multiple streams
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  thread t1(threadFuncDiff_1, stream1, stream2);
  thread t2(threadFuncDiff_2, stream1, stream2);
  t1.join();
  t2.join();
  // Destroy streams
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}
/**
 * End doxygen group StreamTest.
 * @}
 */
