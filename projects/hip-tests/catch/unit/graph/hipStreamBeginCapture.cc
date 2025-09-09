/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"  // NOLINT

#pragma clang diagnostic ignored "-Wunused-variable"
/**
 * @addtogroup hipStreamBeginCapture hipStreamBeginCapture
 * @{
 * @ingroup GraphTest
 * `hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode)` -
 * begins graph capture on a stream
 */

static int gCbackIter = 0;

static __global__ void dummyKernel() { return; }

static __global__ void incrementKernel(int* data) {
  atomicAdd(data, 1);
  return;
}

static __global__ void myadd(int* A_d, int* B_d) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  A_d[myId] = A_d[myId] + B_d[myId];
}

static __global__ void mymul(int* devMem, int value) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  devMem[myId] = devMem[myId] * value;
}

static void hostNodeCallback(void* data) {
  REQUIRE(data == nullptr);
  gCbackIter++;
}

// Functors for testing concurrent stream capture codes
class streamSync {
 public:
  void operator()(hipStream_t stream) { result_status = hipStreamSynchronize(stream); }
  hipError_t result_status = hipSuccess;
};
class streamQuery {
 public:
  void operator()(hipStream_t stream) { result_status = hipStreamQuery(stream); }
  hipError_t result_status = hipSuccess;
};
class deviceSync {
 public:
  void operator()() { result_status = hipDeviceSynchronize(); }
  hipError_t result_status = hipSuccess;
};
class eventSync {
 public:
  void operator()(hipEvent_t event) { auto status = hipEventSynchronize(event); }
  hipError_t result_status = hipSuccess;
};
class eventQuery {
 public:
  void operator()(hipEvent_t event) { auto status = hipEventQuery(event); }
  hipError_t result_status = hipSuccess;
};

template <typename T, typename F>
void captureStreamAndLaunchGraph(F graphFunc, hipStreamCaptureMode mode, hipStream_t stream) {
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(T);

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  // Host and Device allocation
  LinearAllocGuard<T> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<T> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<T> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<T> B_d(LinearAllocs::hipMalloc, Nbytes);

  // Capture stream sequence
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  graphFunc(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream);

  captureSequenceCompute(A_d.ptr(), B_h.ptr(), B_d.ptr(), N, stream);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), N, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i) * static_cast<float>(i), N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Basic Functional Test for capturing created/hipStreamPerThread stream
 * and replaying sequence. Test exercises the API on all available modes:
 *        -# Linear sequence capture - each graph node has only one dependency
 *        -# Branched sequence capture - some graph nodes have more than one
 * dependency
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_Functional") {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  EventsGuard events_guard(3);
  StreamsGuard streams_guard(2);

  SECTION("Linear graph capture") {
    captureStreamAndLaunchGraph<float>(
        [](float* A_h, float* A_d, float* B_h, float* B_d, size_t N, hipStream_t stream) {
          return captureSequenceLinear(A_h, A_d, B_h, B_d, N, stream);
        },
        captureMode, stream);
  }

  SECTION("Branched graph capture") {
    captureStreamAndLaunchGraph<float>(
        [&streams_guard, &events_guard](float* A_h, float* A_d, float* B_h, float* B_d, size_t N,
                                        hipStream_t stream) {
          captureSequenceBranched(A_h, A_d, B_h, B_d, N, stream, streams_guard.stream_list(),
                                  events_guard.event_list());
        },
        captureMode, stream);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Begin capture on legacy/null stream
 *        -# Begin capture on the already captured stream
 *        -# Begin capture with invalid mode
 *        -# Begin capture on uninitialized stream
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Negative_Parameters") {
  const auto stream_type = GENERATE(Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  SECTION("Stream capture on legacy/null stream returns error code.") {
    HIP_CHECK_ERROR(hipStreamBeginCapture(nullptr, hipStreamCaptureModeGlobal),
                    hipErrorStreamCaptureUnsupported);
  }
  SECTION("Capturing hipStream status with same stream again") {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
                    hipErrorIllegalState);
  }
  SECTION("Creating hipStream with invalid mode") {
    HIP_CHECK_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureMode(-1)), hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Basic Test to verify basic API functionality with
 * created/hipStreamPerThread stream for available modes
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_Basic") {
  hipGraph_t graph{nullptr};
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t s = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  HIP_CHECK(hipStreamBeginCapture(s, captureMode));

  HIP_CHECK(hipStreamEndCapture(s, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Local function for inter stream event synchronization
 */
static void interStrmEventSyncCapture(const hipStream_t& stream1, const hipStream_t& stream2) {
  hipGraph_t graph1{nullptr}, graph2{nullptr};
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};

  EventsGuard events_guard(1);
  hipEvent_t event = events_guard[0];

  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(event, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, event, 0));
  dummyKernel<<<1, 1, 0, stream1>>>();
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
  dummyKernel<<<1, 1, 0, stream2>>>();
  dummyKernel<<<1, 1, 0, stream2>>>();
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));

  size_t numNodes1 = 0, numNodes2 = 0;
  HIP_CHECK(hipGraphGetNodes(graph1, nullptr, &numNodes1));
  HIP_CHECK(hipGraphGetNodes(graph2, nullptr, &numNodes2));
  REQUIRE(numNodes1 == 1);
  REQUIRE(numNodes2 == 2);

  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  REQUIRE(graphExec1 != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  REQUIRE(graphExec2 != nullptr);

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    // Execute the Graphs
    HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
    HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));
  }

  // Free
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/* Local function for colligated stream capture
 */
static void colligatedStrmCapture(const hipStream_t& stream1, const hipStream_t& stream2) {
  hipGraph_t graph1{nullptr}, graph2{nullptr};
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};

  EventsGuard events_guard(1);
  hipEvent_t event = events_guard[0];

  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(event, stream1));
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamWaitEvent(stream1, event, 0));
  dummyKernel<<<1, 1, 0, stream1>>>();
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
  dummyKernel<<<1, 1, 0, stream2>>>();
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
  // Validate end capture is successful
  REQUIRE(graph2 != nullptr);
  REQUIRE(graph1 != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  REQUIRE(graphExec1 != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  REQUIRE(graphExec2 != nullptr);

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    // Execute the Graphs
    HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
    HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));
  }

  // Free
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/* Local function for colligated stream capture functionality
 */
static void colligatedStrmCaptureFunc(const hipStream_t& stream1, const hipStream_t& stream2) {
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(int);

  hipGraph_t graph1{nullptr}, graph2{nullptr};
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};

  // Host and device allocation
  LinearAllocGuard<int> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<int> B_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<int> C_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> C_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<int> D_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> D_d(LinearAllocs::hipMalloc, Nbytes);

  // Capture 2 streams
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
  captureSequenceLinear(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream1);
  captureSequenceLinear(C_h.host_ptr(), C_d.ptr(), D_h.host_ptr(), D_d.ptr(), N, stream2);
  captureSequenceCompute(A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream1);
  captureSequenceCompute(C_d.ptr(), D_h.host_ptr(), D_d.ptr(), N, stream2);
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
  // Validate end capture is successful
  REQUIRE(graph2 != nullptr);
  REQUIRE(graph1 != nullptr);

  // Create Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  REQUIRE(graphExec1 != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  REQUIRE(graphExec2 != nullptr);

  // Execute the Graphs
  for (size_t iter = 0; iter < kLaunchIters; iter++) {
    std::fill_n(A_h.host_ptr(), N, iter);
    std::fill_n(C_h.host_ptr(), N, iter);
    HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
    HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<int>(iter * iter), N);
    ArrayFindIfNot(D_h.host_ptr(), static_cast<int>(iter * iter), N);
  }

  // Free
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/* Stream Capture thread function
 */
static void threadStrmCaptureFunc(hipStream_t stream, int* A_h, int* A_d, int* B_h, int* B_d,
                                  hipGraph_t* graph, size_t N, hipStreamCaptureMode mode) {
  // Capture stream
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  captureSequenceLinear(A_h, A_d, B_h, B_d, N, stream);
  captureSequenceCompute(A_d, B_h, B_d, N, stream);
  HIP_CHECK(hipStreamEndCapture(stream, graph));
}

/* Local Function for multithreaded tests
 */
static void multithreadedTest(hipStreamCaptureMode mode) {
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(int);

  hipGraph_t graph1{nullptr}, graph2{nullptr};
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};
  StreamGuard stream_guard1(Streams::created);
  hipStream_t stream1 = stream_guard1.stream();
  StreamGuard stream_guard2(Streams::created);
  hipStream_t stream2 = stream_guard2.stream();

  // Host and device allocation
  LinearAllocGuard<int> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<int> B_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<int> C_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> D_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<int> C_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<int> D_d(LinearAllocs::hipMalloc, Nbytes);

  // Launch 2 threads to capture the 2 streams into graphs
  std::thread t1(threadStrmCaptureFunc, stream1, A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(),
                 B_d.ptr(), &graph1, N, mode);
  std::thread t2(threadStrmCaptureFunc, stream2, C_h.host_ptr(), C_d.ptr(), D_h.host_ptr(),
                 D_d.ptr(), &graph2, N, mode);
  t1.join();
  t2.join();

  // Create Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  REQUIRE(graphExec1 != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  REQUIRE(graphExec2 != nullptr);

  // Execute the Graphs
  for (size_t iter = 0; iter < kLaunchIters; iter++) {
    std::fill_n(A_h.host_ptr(), N, iter);
    std::fill_n(C_h.host_ptr(), N, iter);
    HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
    HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<int>(iter * iter), N);
    ArrayFindIfNot(D_h.host_ptr(), static_cast<int>(iter * iter), N);
  }

  // Free
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify inter stream event synchronization- Waiting on an event
 recorded on a captured stream. Initiate capture on stream1, record an event on
 stream1, wait for the event on stream2, end the stream1 capture and initiate
 stream capture on stream2
 *        -# Streams are created with hipStreamDefault/hipStreamNonBlocking flag
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_InterStrmEventSync_Flags") {
  const auto stream_flags1 = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  const auto stream_flags2 = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  StreamGuard stream_guard1(Streams::withFlags, stream_flags1);
  hipStream_t stream1 = stream_guard1.stream();
  StreamGuard stream_guard2(Streams::withFlags, stream_flags2);
  hipStream_t stream2 = stream_guard2.stream();
  interStrmEventSyncCapture(stream1, stream2);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify inter stream event synchronization- Waiting on an event
 * recorded on a captured stream. Initiate capture on stream1, record an event
 * on stream1, wait for the event on stream2, end the stream1 capture and
 * initiate stream capture on stream2
 *        -# Stream1 is created with minimal priority, stream 2 is created with
 * maximal priority
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_InterStrmEventSync_Priority") {
  int minPriority = 0, maxPriority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
  StreamGuard stream_guard1(Streams::withPriority, hipStreamDefault, minPriority);
  hipStream_t stream1 = stream_guard1.stream();
  StreamGuard stream_guard2(Streams::withPriority, hipStreamDefault, maxPriority);
  hipStream_t stream2 = stream_guard2.stream();
  interStrmEventSyncCapture(stream1, stream2);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify colligated streams capture. Capture operation sequences
 * queued in 2 streams by overlapping the 2 captures. Initiate capture on
 * stream1, record an event on stream1, initiate capture on stream 2, end both
 * stream captures
 *        -# Streams are created with hipStreamDefault/hipStreamNonBlocking flag
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_ColligatedStrmCapture_Flags") {
  const auto stream_flags1 = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  const auto stream_flags2 = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  StreamGuard stream_guard1(Streams::withFlags, stream_flags1);
  hipStream_t stream1 = stream_guard1.stream();
  StreamGuard stream_guard2(Streams::withFlags, stream_flags2);
  hipStream_t stream2 = stream_guard2.stream();
  colligatedStrmCapture(stream1, stream2);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify colligated streams capture. Capture operation sequences
 * queued in 2 streams by overlapping the 2 captures. Initiate capture on
 * stream1, record an event on stream1, initiate capture on stream 2, end both
 * stream captures
 *        -# Stream1 is created with minimal priority, stream 2 is created with
 * maximal priority
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_ColligatedStrmCapture_Prio") {
  int minPriority = 0, maxPriority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
  StreamGuard stream_guard1(Streams::withPriority, hipStreamDefault, minPriority);
  hipStream_t stream1 = stream_guard1.stream();
  StreamGuard stream_guard2(Streams::withPriority, hipStreamDefault, maxPriority);
  hipStream_t stream2 = stream_guard2.stream();
  colligatedStrmCapture(stream1, stream2);
}

/**
 * Test Description
 * ------------------------
 *    - Create 2 streams. Start capturing both stream1 and stream2 at the same
 * time. On stream1 queue memcpy, kernel and memcpy operations and on stream2
 * queue memcpy, kernel and memcpy operations. Execute both the captured graphs
 * and validate the results
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_ColligatedStrmCaptureFunc") {
  StreamGuard stream_guard1(Streams::created);
  hipStream_t stream1 = stream_guard1.stream();
  StreamGuard stream_guard2(Streams::created);
  hipStream_t stream2 = stream_guard2.stream();
  colligatedStrmCaptureFunc(stream1, stream2);
}

/**
 * Test Description
 * ------------------------
 *    - Capture 2 streams in parallel using threads. Execute the graphs in
 * sequence in main thread and validate the results for all available capture
 * modes
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_Multithreaded") {
  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
  multithreadedTest(captureMode);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify inter stream event synchronization- Waiting on an event
 * recorded on a captured stream.
 *        -# Initiate capture on stream1, record an event on stream1, wait for
 * the event on stream2, end the stream1 capture and initiate stream capture on
 * stream2. Repeat the same sequence between stream2 and stream3
 *        -# Initiate capture on stream1, record an event on stream1, wait for
 * the event on stream2 and stream3, end the stream1 capture and initiate stream
 * capture on stream2 and stream3
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_Multiplestrms") {
  StreamsGuard streams(3);
  hipGraph_t graphs[3];

  size_t numNodes1 = 0, numNodes2 = 0, numNodes3 = 0;
  SECTION("Capture Multiple stream with interdependent events") {
    EventsGuard events(2);

    HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipEventRecord(events[0], streams[0]));
    HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
    dummyKernel<<<1, 1, 0, streams[0]>>>();
    HIP_CHECK(hipStreamEndCapture(streams[0], &graphs[0]));
    HIP_CHECK(hipStreamBeginCapture(streams[1], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipEventRecord(events[1], streams[1]));
    HIP_CHECK(hipStreamWaitEvent(streams[2], events[1], 0));
    dummyKernel<<<1, 1, 0, streams[1]>>>();
    HIP_CHECK(hipStreamEndCapture(streams[1], &graphs[1]));
    HIP_CHECK(hipStreamBeginCapture(streams[2], hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, streams[2]>>>();
    HIP_CHECK(hipStreamEndCapture(streams[2], &graphs[2]));
    HIP_CHECK(hipGraphGetNodes(graphs[0], nullptr, &numNodes1));
    HIP_CHECK(hipGraphGetNodes(graphs[1], nullptr, &numNodes2));
    HIP_CHECK(hipGraphGetNodes(graphs[2], nullptr, &numNodes3));
    REQUIRE(numNodes1 == 1);
    REQUIRE(numNodes2 == 1);
    REQUIRE(numNodes3 == 1);
  }
  SECTION("Capture Multiple stream with single event") {
    EventsGuard events(1);
    hipEvent_t event = events[0];

    HIP_CHECK(hipEventCreate(&event));
    HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipEventRecord(event, streams[0]));
    HIP_CHECK(hipStreamWaitEvent(streams[1], event, 0));
    HIP_CHECK(hipStreamWaitEvent(streams[2], event, 0));
    dummyKernel<<<1, 1, 0, streams[0]>>>();
    HIP_CHECK(hipStreamEndCapture(streams[0], &graphs[0]));
    HIP_CHECK(hipStreamBeginCapture(streams[1], hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, streams[1]>>>();
    HIP_CHECK(hipStreamEndCapture(streams[1], &graphs[1]));
    HIP_CHECK(hipStreamBeginCapture(streams[2], hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, streams[2]>>>();
    HIP_CHECK(hipStreamEndCapture(streams[2], &graphs[2]));
    HIP_CHECK(hipGraphGetNodes(graphs[0], nullptr, &numNodes1));
    HIP_CHECK(hipGraphGetNodes(graphs[1], nullptr, &numNodes2));
    HIP_CHECK(hipGraphGetNodes(graphs[2], nullptr, &numNodes3));
    REQUIRE(numNodes1 == 1);
    REQUIRE(numNodes2 == 1);
    REQUIRE(numNodes3 == 1);
  }

  for (int i = 0; i < 3; i++) {
    HIP_CHECK(hipGraphDestroy(graphs[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify queue operations (increment kernels) in 3 streams. Start
 * capturing the streams after some operations have been queued. This scenario
 * validates that only operations queued after hipStreamBeginCapture are
 * captured in the graph
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_CapturingFromWithinStrms") {
  constexpr int INCREMENT_KERNEL_FINALEXP_VAL = 7;

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  StreamsGuard streams(3);
  EventsGuard events(3);

  // Create a device memory of size int and initialize it to 0
  LinearAllocGuard<int> hostMem_g(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> devMem_g(LinearAllocs::hipMalloc, sizeof(int));
  int* hostMem = hostMem_g.host_ptr();
  int* devMem = devMem_g.ptr();
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Start Capturing
  incrementKernel<<<1, 1, 0, streams[0]>>>(devMem);
  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem);
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[2], events[0], 0));
  incrementKernel<<<1, 1, 0, streams[0]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[0]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem);
  HIP_CHECK(hipEventRecord(events[1], streams[1]));
  HIP_CHECK(hipEventRecord(events[2], streams[2]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[1], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[2], 0));
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, sizeof(int), hipMemcpyDefault, streams[0]));
  HIP_CHECK(hipStreamEndCapture(streams[0], &graph));  // End Capture
  // Reset device memory
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());

  // Create Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  HIP_CHECK(hipGraphLaunch(graphExec, streams[0]));
  HIP_CHECK(hipStreamSynchronize(streams[0]));
  REQUIRE((*hostMem) == INCREMENT_KERNEL_FINALEXP_VAL);

  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Detecting invalid capture. Create 2 streams s1 and s2. Start capturing
 * s1. Create event dependency between s1 and s2 using event record and event
 * wait. Try capturing s2. hipStreamBeginCapture must return error
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Negative_DetectingInvalidCapture") {
  StreamsGuard streams(2);
  EventsGuard events(1);
  hipEvent_t event = events[0];
  hipGraph_t graph;
  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(event, streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], event, 0));
  dummyKernel<<<1, 1, 0, streams[0]>>>();
  // Since stream[1] is already in capture mode due to event wait
  // hipStreamBeginCapture on stream[1] is expected to return error.
  HIP_CHECK_ERROR(hipStreamBeginCapture(streams[1], hipStreamCaptureModeGlobal),
                  hipErrorIllegalState);
  HIP_CHECK(hipStreamEndCapture(streams[0], &graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify wtream reuse. Capture multiple graphs from the same
 * stream. Validate graphs are captured correctly
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_CapturingMultGraphsFrom1Strm") {
  hipGraph_t graphs[3];

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream1 = stream_guard.stream();

  // Create a device memory of size int and initialize it to 0
  LinearAllocGuard<int> hostMem_g(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> devMem_g(LinearAllocs::hipMalloc, sizeof(int));
  int* hostMem = hostMem_g.host_ptr();
  int* devMem = devMem_g.ptr();
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < 3; i++) {
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
    for (int j = 0; j <= i; j++) incrementKernel<<<1, 1, 0, stream1>>>(devMem);
    HIP_CHECK(hipMemcpyAsync(hostMem, devMem, sizeof(int), hipMemcpyDefault, stream1));
    HIP_CHECK(hipStreamEndCapture(stream1, &graphs[i]));
  }
  // Instantiate and execute all graphs
  for (int i = 0; i < 3; i++) {
    hipGraphExec_t graphExec{nullptr};
    HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graphs[i], nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));
    REQUIRE((*hostMem) == (i + 1));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graphs[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify synchronization during stream capture returns an error:
 *        -# Synchronize stream during capture
 *        -# Synchronize device during capture
 *        -# Synchronize event during capture
 *        -# Query stream during capture
 *        -# Query for an event during capture
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Negative_CheckingSyncDuringCapture") {
  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
  const unsigned int stream_flag = GENERATE(hipStreamDefault, hipStreamNonBlocking);

  StreamGuard stream_guard(Streams::created, stream_flag);
  hipStream_t stream = stream_guard.stream();

  EventsGuard events_guard(1);
  hipEvent_t e = events_guard[0];

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  SECTION("Synchronize stream during capture") {
    HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorStreamCaptureUnsupported);
  }
  SECTION("Query stream during capture") {
    HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorStreamCaptureUnsupported);
  }
  SECTION("Synchronize device during capture") {
    HIP_CHECK_ERROR(hipDeviceSynchronize(), hipErrorStreamCaptureUnsupported);
  }
  SECTION("Synchronize event during capture") {
    HIP_CHECK(hipEventRecord(e, stream));
    HIP_CHECK_ERROR(hipEventSynchronize(e), hipErrorCapturedEvent);
  }
  SECTION("Query for an event during capture") {
    HIP_CHECK(hipEventRecord(e, stream));
    HIP_CHECK_ERROR(hipEventQuery(e), hipErrorCapturedEvent);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify synchronization during stream capture returns an error:
 *        -# Synchronize stream during capture
 *        -# Synchronize device during capture
 *        -# Synchronize event during capture
 *        -# Query stream during capture
 *        -# Query for an event during capture
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Negative_Concurrent_CheckingSyncDuringCapture") {
  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
  const unsigned int stream_flag = GENERATE(hipStreamDefault, hipStreamNonBlocking);

  StreamGuard stream_guard(Streams::created, stream_flag);
  StreamGuard concurrent_stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();
  hipStream_t concurrent_stream = concurrent_stream_guard.stream();

  EventsGuard events_guard(1);
  hipEvent_t e = events_guard[0];

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  SECTION("Synchronize stream during capture") {
    streamSync func;
    hipError_t expected = hipSuccess;
    if (captureMode == hipStreamCaptureModeGlobal) expected = hipErrorStreamCaptureUnsupported;

    std::thread t(std::ref(func), concurrent_stream);
    t.join();
    REQUIRE(func.result_status == expected);
  }
  SECTION("Query stream during capture") {
    streamQuery func;
    hipError_t expected = hipSuccess;
    if (captureMode == hipStreamCaptureModeGlobal) expected = hipErrorStreamCaptureUnsupported;

    std::thread t(std::ref(func), concurrent_stream);
    t.join();
    REQUIRE(func.result_status == expected);
  }
  SECTION("Synchronize device during capture") {
    deviceSync func;
    hipError_t expected = hipErrorStreamCaptureUnsupported;

    std::thread t(std::ref(func));
    t.join();
    REQUIRE(func.result_status == expected);
  }
  SECTION("Synchronize event during capture") {
    eventSync func;
    hipError_t expected = hipSuccess;

    std::thread t(std::ref(func), e);
    t.join();
    REQUIRE(func.result_status == expected);
  }
  SECTION("Query for an event during capture") {
    eventQuery func;
    hipError_t expected = hipSuccess;

    std::thread t(std::ref(func), e);
    t.join();
    REQUIRE(func.result_status == expected);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify unsafe API calls during stream capture with initiated
 * with hipStreamCaptureModeGlobal and hipStreamCaptureModeThreadLocal return an
 * error:
 *        -# hipMalloc during capture
 *        -# hipMemcpy during capture
 *        -# hipMemset during capture
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Negative_UnsafeCallsDuringCapture") {
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  LinearAllocGuard<int> hostMem(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> devMem(LinearAllocs::hipMalloc, sizeof(int));

  int* devMem2;

  const hipStreamCaptureMode captureMode =
      GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal);

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  SECTION("hipMalloc during capture") {
    HIP_CHECK_ERROR(hipMalloc(&devMem2, sizeof(int)), hipErrorStreamCaptureUnsupported);
  }
  SECTION("hipMemcpy during capture") {
    HIP_CHECK_ERROR(hipMemcpy(devMem.ptr(), hostMem.host_ptr(), sizeof(int), hipMemcpyHostToDevice),
                    hipErrorStreamCaptureImplicit);
  }
  SECTION("hipMemset during capture") {
    HIP_CHECK_ERROR(hipMemset(devMem.ptr(), 0, sizeof(int)), hipErrorStreamCaptureImplicit);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify end stream capture when the stream capture is still in
 * progress:
 *        -# Abruptly end stream capture when stream capture is in progress in
 * forked stream. hipStreamEndCapture must return an error
 *        -# Abruptly end stream capture when operations in forked stream are
 * still waiting to be captured. hipStreamEndCapture must return an error
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Negative_EndingCapwhenCapInProg") {
  hipGraph_t graph{nullptr};

  StreamsGuard streams_guard(2);
  hipStream_t stream1 = streams_guard[0];
  hipStream_t stream2 = streams_guard[1];

  SECTION("Abruptly end strm capture when in progress in forked strm") {
    EventsGuard events_guard(1);
    hipEvent_t e = events_guard[0];
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, stream1>>>();
    HIP_CHECK(hipEventRecord(e, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, e, 0));
    dummyKernel<<<1, 1, 0, stream2>>>();
    HIP_CHECK_ERROR(hipStreamEndCapture(stream1, &graph), hipErrorStreamCaptureUnjoined);
  }
  SECTION("End strm capture when forked strm still has operations") {
    EventsGuard events_guard(2);
    hipEvent_t e1 = events_guard[0];
    hipEvent_t e2 = events_guard[1];
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, stream1>>>();
    HIP_CHECK(hipEventRecord(e1, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
    dummyKernel<<<1, 1, 0, stream2>>>();
    HIP_CHECK(hipEventRecord(e2, stream2));
    HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
    dummyKernel<<<1, 1, 0, stream2>>>();
    HIP_CHECK_ERROR(hipStreamEndCapture(stream1, &graph), hipErrorStreamCaptureUnjoined);
  }
}
/**
 * Test Description
 * ------------------------
 *    - Testing independent stream capture using multiple GPUs. Capture a stream
 * in each device context and execute the captured graph in the context GPU
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_MultiGPU") {
  int devcount = 0;
  HIP_CHECK(hipGetDeviceCount(&devcount));
  // If only single GPU is detected then return
  if (devcount < 2) {
    SUCCEED("skipping the testcases as numDevices < 2");
    return;
  }
  hipStream_t* stream = reinterpret_cast<hipStream_t*>(malloc(devcount * sizeof(hipStream_t)));
  REQUIRE(stream != nullptr);
  hipGraph_t* graph = reinterpret_cast<hipGraph_t*>(malloc(devcount * sizeof(hipGraph_t)));
  REQUIRE(graph != nullptr);
  int **devMem{nullptr}, **hostMem{nullptr};
  hostMem = reinterpret_cast<int**>(malloc(sizeof(int*) * devcount));
  REQUIRE(hostMem != nullptr);
  devMem = reinterpret_cast<int**>(malloc(sizeof(int*) * devcount));
  REQUIRE(devMem != nullptr);
  hipGraphExec_t* graphExec =
      reinterpret_cast<hipGraphExec_t*>(malloc(devcount * sizeof(hipGraphExec_t)));
  // Capture stream in each device
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&stream[dev]));
    hostMem[dev] = reinterpret_cast<int*>(malloc(sizeof(int)));
    HIP_CHECK(hipMalloc(&devMem[dev], sizeof(int)));
    HIP_CHECK(hipStreamBeginCapture(stream[dev], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemsetAsync(devMem[dev], 0, sizeof(int), stream[dev]));
    for (int i = 0; i < (dev + 1); i++) {
      incrementKernel<<<1, 1, 0, stream[dev]>>>(devMem[dev]);
    }
    HIP_CHECK(
        hipMemcpyAsync(hostMem[dev], devMem[dev], sizeof(int), hipMemcpyDefault, stream[dev]));
    HIP_CHECK(hipStreamEndCapture(stream[dev], &graph[dev]));
  }
  // Launch the captured graphs in the respective device
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipGraphInstantiate(&graphExec[dev], graph[dev], nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec[dev], stream[dev]));
  }
  // Validate output
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(stream[dev]));
    REQUIRE((*hostMem[dev]) == (dev + 1));
  }
  // Destroy all device resources
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipGraphExecDestroy(graphExec[dev]));
    HIP_CHECK(hipStreamDestroy(stream[dev]));
  }
  free(graphExec);
  free(hostMem);
  free(devMem);
  free(stream);
  free(graph);
}

/**
 * Test Description
 * ------------------------
 *    - Test Nested Stream Capture Functionality: Create 3 streams. Capture s1,
 * record event e1 on s1, wait for event e1 on s2 and queue operations in s1.
 * Record event e2 on s2 and wait for it on s3. Queue operations on both s2 and
 * s3. Record event e4 on s3 and wait for it in s1. Record event e3 on s2 and
 * wait for it in s1. End stream capture on s1. Execute the graph and verify the
 * result.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_nestedStreamCapture") {
  constexpr int INCREMENT_KERNEL_FINALEXP_VAL = 7;

  hipGraph_t graph{nullptr};
  StreamsGuard streams(3);
  EventsGuard events(4);

  // Create a device memory of size int and initialize it to 0
  LinearAllocGuard<int> hostMem_g(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> devMem_g(LinearAllocs::hipMalloc, sizeof(int));
  HIP_CHECK(hipMemset(devMem_g.ptr(), 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Start Capturing stream1
  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(hipEventRecord(events[1], streams[1]));
  HIP_CHECK(hipStreamWaitEvent(streams[2], events[1], 0));
  incrementKernel<<<1, 1, 0, streams[0]>>>(devMem_g.ptr());
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem_g.ptr());
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem_g.ptr());
  incrementKernel<<<1, 1, 0, streams[0]>>>(devMem_g.ptr());
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem_g.ptr());
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem_g.ptr());
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem_g.ptr());
  HIP_CHECK(hipEventRecord(events[2], streams[1]));
  HIP_CHECK(hipEventRecord(events[3], streams[2]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[3], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[2], 0));
  HIP_CHECK(hipMemcpyAsync(hostMem_g.host_ptr(), devMem_g.ptr(), sizeof(int), hipMemcpyDefault,
                           streams[0]));
  HIP_CHECK(hipStreamEndCapture(streams[0], &graph));  // End Capture
  // Reset device memory
  HIP_CHECK(hipMemset(devMem_g.ptr(), 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Create Executable Graphs
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streams[0]));
  HIP_CHECK(hipStreamSynchronize(streams[0]));
  REQUIRE((*hostMem_g.host_ptr()) == INCREMENT_KERNEL_FINALEXP_VAL);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test Nested Stream Capture Functionality: Create 3 streams. Capture s1,
 * record event e1 on s1, wait for event e1 on s2 and queue operations in s1.
 * Record event e2 on s2 and wait for it on s3. Queue operations on both s2 and
 * s3. Record event e4 on s3 and wait for it in s1. Record event e3 on s2 and
 * wait for it in s1. End stream capture on s1. Queue operations on both s2 and
 * s3, and capture their graphs. Execute the graphs and verify the result.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_streamReuse") {
  constexpr int increment_kernel_vals[3] = {7, 3, 5};

  hipGraph_t graphs[3];
  StreamsGuard streams(3);
  EventsGuard events(4);
  LinearAllocGuard<int> hostMem_g1 = LinearAllocGuard<int>(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> hostMem_g2 = LinearAllocGuard<int>(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> hostMem_g3 = LinearAllocGuard<int>(LinearAllocs::malloc, sizeof(int));
  LinearAllocGuard<int> devMem_g1 = LinearAllocGuard<int>(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> devMem_g2 = LinearAllocGuard<int>(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> devMem_g3 = LinearAllocGuard<int>(LinearAllocs::hipMalloc, sizeof(int));

  std::vector<int*> hostMem = {hostMem_g1.host_ptr(), hostMem_g2.host_ptr(), hostMem_g3.host_ptr()};
  std::vector<int*> devMem = {devMem_g1.ptr(), devMem_g2.ptr(), devMem_g3.ptr()};
  // Create a device memory of size int and initialize it to 0
  for (int i = 0; i < 3; i++) {
    memset(hostMem[i], 0, sizeof(int));
    HIP_CHECK(hipMemset(devMem[i], 0, sizeof(int)));
  }
  HIP_CHECK(hipDeviceSynchronize());
  // Start Capturing stream1
  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(hipEventRecord(events[1], streams[1]));
  HIP_CHECK(hipStreamWaitEvent(streams[2], events[1], 0));
  incrementKernel<<<1, 1, 0, streams[0]>>>(devMem[0]);
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem[0]);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[0]);
  incrementKernel<<<1, 1, 0, streams[0]>>>(devMem[0]);
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem[0]);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[0]);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[0]);
  HIP_CHECK(hipEventRecord(events[2], streams[1]));
  HIP_CHECK(hipEventRecord(events[3], streams[2]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[3], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[2], 0));
  HIP_CHECK(hipMemcpyAsync(hostMem[0], devMem[0], sizeof(int), hipMemcpyDefault, streams[0]));
  HIP_CHECK(hipStreamEndCapture(streams[0], &graphs[0]));  // End Capture
  // Start capturing graph2 from stream 2
  HIP_CHECK(hipStreamBeginCapture(streams[1], hipStreamCaptureModeGlobal));
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem[1]);
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem[1]);
  incrementKernel<<<1, 1, 0, streams[1]>>>(devMem[1]);
  HIP_CHECK(hipMemcpyAsync(hostMem[1], devMem[1], sizeof(int), hipMemcpyDefault, streams[1]));
  HIP_CHECK(hipStreamEndCapture(streams[1], &graphs[1]));  // End Capture
  // Start capturing graph3 from stream 3
  HIP_CHECK(hipStreamBeginCapture(streams[2], hipStreamCaptureModeGlobal));
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[2]);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[2]);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[2]);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[2]);
  incrementKernel<<<1, 1, 0, streams[2]>>>(devMem[2]);
  HIP_CHECK(hipMemcpyAsync(hostMem[2], devMem[2], sizeof(int), hipMemcpyDefault, streams[2]));
  HIP_CHECK(hipStreamEndCapture(streams[2], &graphs[2]));  // End Capture
  // Reset device memory
  HIP_CHECK(hipMemset(devMem[0], 0, sizeof(int)));
  HIP_CHECK(hipMemset(devMem[1], 0, sizeof(int)));
  HIP_CHECK(hipMemset(devMem[2], 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Create Executable Graphs and verify graphs
  for (int i = 0; i < 3; i++) {
    hipGraphExec_t graphExec{nullptr};
    HIP_CHECK(hipMemset(devMem[i], 0, sizeof(int)));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graphs[i], nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, streams[i]));
    HIP_CHECK(hipStreamSynchronize(streams[i]));
    REQUIRE((*hostMem[i]) == increment_kernel_vals[i]);
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graphs[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Capture a complex graph containing multiple independent memcpy, kernel
 * and host nodes. Launch the graph on random input data and validate the output
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_captureComplexGraph") {
  constexpr int GRIDSIZE = 256;
  constexpr int BLOCKSIZE = 256;
  constexpr int CONST_KER1_VAL = 3;
  constexpr int CONST_KER2_VAL = 2;
  constexpr int CONST_KER3_VAL = 5;

  hipGraph_t graph{nullptr};
  StreamsGuard streams(5);
  EventsGuard events(7);
  // Allocate Device memory and Host memory
  size_t N = GRIDSIZE * BLOCKSIZE;
  LinearAllocGuard<int> Ah = LinearAllocGuard<int>(LinearAllocs::malloc, N * sizeof(int));
  LinearAllocGuard<int> Bh = LinearAllocGuard<int>(LinearAllocs::malloc, N * sizeof(int));
  LinearAllocGuard<int> Ch = LinearAllocGuard<int>(LinearAllocs::malloc, N * sizeof(int));
  LinearAllocGuard<int> Ad = LinearAllocGuard<int>(LinearAllocs::hipMalloc, N * sizeof(int));
  LinearAllocGuard<int> Bd = LinearAllocGuard<int>(LinearAllocs::hipMalloc, N * sizeof(int));

  // Capture streams into graph
  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[3], events[0], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[4], events[0], 0));
  HIP_CHECK(
      hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), (N * sizeof(int)), hipMemcpyDefault, streams[0]));
  HIP_CHECK(
      hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), (N * sizeof(int)), hipMemcpyDefault, streams[4]));
  hipHostFn_t fn = hostNodeCallback;
  HIPCHECK(hipLaunchHostFunc(streams[3], fn, nullptr));
  HIP_CHECK(hipEventRecord(events[1], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[1], 0));
  int* Ad_2nd_half = Ad.ptr() + N / 2;
  int* Ad_1st_half = Ad.ptr();
  mymul<<<GRIDSIZE / 2, BLOCKSIZE, 0, streams[0]>>>(Ad_2nd_half, CONST_KER2_VAL);
  mymul<<<GRIDSIZE / 2, BLOCKSIZE, 0, streams[1]>>>(Ad_1st_half, CONST_KER1_VAL);
  HIP_CHECK(hipEventRecord(events[2], streams[1]));
  HIP_CHECK(hipStreamWaitEvent(streams[2], events[2], 0));
  mymul<<<GRIDSIZE / 2, BLOCKSIZE, 0, streams[1]>>>(Ad_1st_half, CONST_KER3_VAL);
  HIPCHECK(hipLaunchHostFunc(streams[2], fn, nullptr));
  HIP_CHECK(hipEventRecord(events[6], streams[1]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[6], 0));
  HIP_CHECK(hipEventRecord(events[5], streams[4]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[5], 0));
  myadd<<<GRIDSIZE, BLOCKSIZE, 0, streams[0]>>>(Ad.ptr(), Bd.ptr());
  HIP_CHECK(hipEventRecord(events[3], streams[2]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[3], 0));
  HIP_CHECK(hipEventRecord(events[4], streams[3]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[4], 0));
  HIP_CHECK(
      hipMemcpyAsync(Ch.host_ptr(), Ad.ptr(), (N * sizeof(int)), hipMemcpyDefault, streams[0]));
  HIP_CHECK(hipStreamEndCapture(streams[0], &graph));  // End Capture
  // Execute and test the graph
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  // Verify graph
  for (size_t iter = 0; iter < kLaunchIters; iter++) {
    std::fill_n(Ah.host_ptr(), N, iter);
    std::fill_n(Bh.host_ptr(), N, iter);
    HIP_CHECK(hipGraphLaunch(graphExec, streams[0]));
    HIP_CHECK(hipStreamSynchronize(streams[0]));
    for (size_t i = 0; i < N; i++) {
      if (i > (N / 2 - 1)) {
        REQUIRE(Ch.host_ptr()[i] == (Bh.host_ptr()[i] + Ah.host_ptr()[i] * CONST_KER2_VAL));
      } else {
        REQUIRE(Ch.host_ptr()[i] ==
                (Bh.host_ptr()[i] + Ah.host_ptr()[i] * CONST_KER1_VAL * CONST_KER3_VAL));
      }
    }
  }
  REQUIRE(gCbackIter == (2 * kLaunchIters));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify capturing empty streams (parent + forked streams) and
 * validate the captured graph has no nodes
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Positive_captureEmptyStreams") {
  hipGraph_t graph{nullptr};

  // Stream and event create
  StreamsGuard streams(3);
  EventsGuard events(3);

  // Capture streams into graph
  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[2], events[0], 0));
  HIP_CHECK(hipEventRecord(events[1], streams[1]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[1], 0));
  HIP_CHECK(hipEventRecord(events[2], streams[2]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[2], 0));
  HIP_CHECK(hipStreamEndCapture(streams[0], &graph));  // End Capture
  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(numNodes == 0);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify hipStreamSynchronize on a stream works when stream capture
 * on another stream is ongoing.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamBeginCapture_StreamSync_OngoingCapture") {
  hipStreamCaptureMode flag = hipStreamCaptureModeRelaxed;
  constexpr int GRIDSIZE = 1;
  constexpr int BLOCKSIZE = 512;
  constexpr int VALUE1 = 7, VALUE2 = 11;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  // Allocate device memory
  LinearAllocGuard<int> Ah = LinearAllocGuard<int>(LinearAllocs::malloc, BLOCKSIZE * sizeof(int));
  LinearAllocGuard<int> Ad =
      LinearAllocGuard<int>(LinearAllocs::hipMalloc, BLOCKSIZE * sizeof(int));
  LinearAllocGuard<int> Bh = LinearAllocGuard<int>(LinearAllocs::malloc, BLOCKSIZE * sizeof(int));
  LinearAllocGuard<int> Bd =
      LinearAllocGuard<int>(LinearAllocs::hipMalloc, BLOCKSIZE * sizeof(int));
  // Fill input data
  std::fill_n(Ah.host_ptr(), BLOCKSIZE, VALUE1);
  std::fill_n(Bh.host_ptr(), BLOCKSIZE, VALUE2);
  // Stream create
  StreamsGuard stream0(1);
  // Capture streams into graph
  SECTION("Stream Creation Before Capture") {
    StreamsGuard stream1(1);
    HIP_CHECK(hipStreamBeginCapture(stream0[0], flag));
    HIP_CHECK(hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipStreamSynchronize(stream1[0]));
    myadd<<<GRIDSIZE, BLOCKSIZE, 0, stream0[0]>>>(Ad.ptr(), Bd.ptr());
    HIP_CHECK(hipStreamEndCapture(stream0[0], &graph));  // End Capture
  }
  SECTION("Synchronizing multiple streams during Capture") {
    StreamsGuard stream1(1), stream2(1);
    HIP_CHECK(hipStreamBeginCapture(stream0[0], flag));
    HIP_CHECK(hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream2[0]));
    HIP_CHECK(hipStreamSynchronize(stream1[0]));
    HIP_CHECK(hipStreamSynchronize(stream2[0]));
    myadd<<<GRIDSIZE, BLOCKSIZE, 0, stream0[0]>>>(Ad.ptr(), Bd.ptr());
    HIP_CHECK(hipStreamEndCapture(stream0[0], &graph));  // End Capture
  }
  SECTION("Stream Creation After Capture") {
    HIP_CHECK(hipStreamBeginCapture(stream0[0], flag));
    StreamsGuard stream1(1);
    HIP_CHECK(hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipStreamSynchronize(stream1[0]));
    myadd<<<GRIDSIZE, BLOCKSIZE, 0, stream0[0]>>>(Ad.ptr(), Bd.ptr());
    HIP_CHECK(hipStreamEndCapture(stream0[0], &graph));  // End Capture
  }
  SECTION("Stream Synchronize Before Capture") {
    StreamsGuard stream1(1);
    HIP_CHECK(hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipStreamSynchronize(stream1[0]));
    HIP_CHECK(hipStreamBeginCapture(stream0[0], flag));
    myadd<<<GRIDSIZE, BLOCKSIZE, 0, stream0[0]>>>(Ad.ptr(), Bd.ptr());
    HIP_CHECK(hipStreamEndCapture(stream0[0], &graph));  // End Capture
  }
  SECTION("Stream Synchronize After Capture") {
    HIP_CHECK(hipStreamBeginCapture(stream0[0], flag));
    myadd<<<GRIDSIZE, BLOCKSIZE, 0, stream0[0]>>>(Ad.ptr(), Bd.ptr());
    HIP_CHECK(hipStreamEndCapture(stream0[0], &graph));  // End Capture
    StreamsGuard stream1(1);
    HIP_CHECK(hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream1[0]));
    HIP_CHECK(hipStreamSynchronize(stream1[0]));
  }
  // Execute and test the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream0[0]));
  HIP_CHECK(hipStreamSynchronize(stream0[0]));
  // Check output
  HIP_CHECK(hipMemcpy(Ah.host_ptr(), Ad.ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDeviceToHost));
  for (int idx = 0; idx < BLOCKSIZE; idx++) {
    REQUIRE(Ah.host_ptr()[idx] == (VALUE1 + VALUE2));
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify hipStreamSynchronize on a stream behavior when stream capture
 * on another stream is ongoing in another thread.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
// Local function executed as thread
static void strmSyncThread(int* Ah, int* Ad, int* Bh, int* Bd, int BLOCKSIZE, hipError_t* error) {
  StreamsGuard stream(1);
  HIP_CHECK(hipMemcpyAsync(Ad, Ah, BLOCKSIZE * sizeof(int), hipMemcpyDefault, stream[0]));
  HIP_CHECK(hipMemcpyAsync(Bd, Bh, BLOCKSIZE * sizeof(int), hipMemcpyDefault, stream[0]));
  *error = hipStreamSynchronize(stream[0]);
}

// Local function executed as thread
static void captureStrmThread(hipGraph_t* graph, int* Ah, int* Ad, int* Bh, int* Bd, int BLOCKSIZE,
                              int GRIDSIZE, hipStreamCaptureMode flag, hipError_t* error) {
  StreamsGuard stream(1);
  // Capture streams into graph
  HIP_CHECK(hipStreamBeginCapture(stream[0], flag));
  std::thread t1(strmSyncThread, Ah, Ad, Bh, Bd, BLOCKSIZE, error);
  t1.join();
  myadd<<<GRIDSIZE, BLOCKSIZE, 0, stream[0]>>>(Ad, Bd);
  if (flag == hipStreamCaptureModeGlobal) {
    HIP_CHECK_ERROR(hipStreamEndCapture(stream[0], graph),
                    hipErrorStreamCaptureInvalidated);  // End Capture
  } else {
    HIP_CHECK(hipStreamEndCapture(stream[0], graph));  // End Capture
  }
}

TEST_CASE("Unit_hipStreamBeginCapture_StreamSync_OngoingCapture_MThread") {
  constexpr int GRIDSIZE = 1;
  constexpr int BLOCKSIZE = 512;
  constexpr int VALUE1 = 7, VALUE2 = 11;
  hipGraph_t graph{nullptr};
  // Allocate device memory
  LinearAllocGuard<int> Ah = LinearAllocGuard<int>(LinearAllocs::malloc, BLOCKSIZE * sizeof(int));
  LinearAllocGuard<int> Ad =
      LinearAllocGuard<int>(LinearAllocs::hipMalloc, BLOCKSIZE * sizeof(int));
  LinearAllocGuard<int> Bh = LinearAllocGuard<int>(LinearAllocs::malloc, BLOCKSIZE * sizeof(int));
  LinearAllocGuard<int> Bd =
      LinearAllocGuard<int>(LinearAllocs::hipMalloc, BLOCKSIZE * sizeof(int));
  // Fill input data
  std::fill_n(Ah.host_ptr(), BLOCKSIZE, VALUE1);
  std::fill_n(Bh.host_ptr(), BLOCKSIZE, VALUE2);
  // Stream create
  hipError_t error = hipSuccess;
  SECTION("Capture Flag = hipStreamCaptureModeGlobal Single Threaded") {
    StreamsGuard stream(2);
    // Capture streams into graph
    HIP_CHECK(hipStreamBeginCapture(stream[0], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream[1]));
    HIP_CHECK(hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream[1]));
    error = hipStreamSynchronize(stream[1]);
    REQUIRE(error == hipErrorStreamCaptureUnsupported);
  }
  SECTION("Capture Flag = hipStreamCaptureModeThreadLocal Single Threaded") {
    StreamsGuard stream(2);
    // Capture streams into graph
    HIP_CHECK(hipStreamBeginCapture(stream[0], hipStreamCaptureModeThreadLocal));
    HIP_CHECK(hipMemcpyAsync(Ad.ptr(), Ah.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream[1]));
    HIP_CHECK(hipMemcpyAsync(Bd.ptr(), Bh.host_ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDefault,
                             stream[1]));
    error = hipStreamSynchronize(stream[1]);
    REQUIRE(error == hipErrorStreamCaptureUnsupported);
  }
  SECTION("Capture Flag = hipStreamCaptureModeGlobal Multithreaded") {
    captureStrmThread(&graph, Ah.host_ptr(), Ad.ptr(), Bh.host_ptr(), Bd.ptr(), BLOCKSIZE, GRIDSIZE,
                      hipStreamCaptureModeGlobal, &error);
    REQUIRE(error == hipErrorStreamCaptureUnsupported);
  }
  SECTION("Capture Flag = hipStreamCaptureModeThreadLocal Multithreaded") {
    captureStrmThread(&graph, Ah.host_ptr(), Ad.ptr(), Bh.host_ptr(), Bd.ptr(), BLOCKSIZE, GRIDSIZE,
                      hipStreamCaptureModeThreadLocal, &error);
    REQUIRE(error == hipSuccess);
  }
  SECTION("Capture Flag = hipStreamCaptureModeRelaxed Multithreaded") {
    captureStrmThread(&graph, Ah.host_ptr(), Ad.ptr(), Bh.host_ptr(), Bd.ptr(), BLOCKSIZE, GRIDSIZE,
                      hipStreamCaptureModeRelaxed, &error);
    REQUIRE(error == hipSuccess);
  }
  if (graph != nullptr) {
    hipGraphExec_t graphExec{nullptr};
    StreamsGuard stream(1);
    // Execute and test the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
    HIP_CHECK(hipStreamSynchronize(stream[0]));
    // Check output
    HIP_CHECK(hipMemcpy(Ah.host_ptr(), Ad.ptr(), BLOCKSIZE * sizeof(int), hipMemcpyDeviceToHost));
    for (int idx = 0; idx < BLOCKSIZE; idx++) {
      REQUIRE(Ah.host_ptr()[idx] == (VALUE1 + VALUE2));
    }
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
