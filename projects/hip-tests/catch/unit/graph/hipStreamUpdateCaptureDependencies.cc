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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"

/**
 * @addtogroup hipStreamUpdateCaptureDependencies
 * hipStreamUpdateCaptureDependencies
 * @{
 * @ingroup GraphTest
 * `hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t
 * *dependencies, size_t numDependencies, unsigned int flags __dparm(0)))` -
 * update the set of dependencies in a capturing stream
 */

static __global__ void vectorSet(const float* A_d, float* B_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    B_d[i] = A_d[i];
  }
}

static __global__ void vectorSum(const float* A_d, const float* B_d, float* C_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    C_d[i] = A_d[i] + B_d[i] + C_d[i];
  }
}

/* Local Function for setting new dependency
 */
static void UpdateStreamCaptureDependenciesSet(hipStream_t stream,
                                               hipStreamCaptureMode captureMode) {
  constexpr size_t N = 1000000;
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(float);

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  hipGraph_t capInfoGraph{nullptr};
  const hipGraphNode_t* nodelist{};
  size_t numDependencies;

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> C_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> B_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> C_d(LinearAllocs::hipMalloc, Nbytes);

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  EventsGuard events_guard(3);
  StreamsGuard streams_guard(2);

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceBranched(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream,
                          streams_guard.stream_list(), events_guard.event_list());

  constexpr int numDepsCreated = 2;  // Num of dependencies created

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capInfoGraph, &nodelist,
                                       &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capInfoGraph != nullptr);
  REQUIRE(numDependencies == numDepsCreated);

  SECTION("Set dependency to independent Memcpy node") {
    // Create memcpy node and set it as a capture dependency in graph
    hipMemcpy3DParms myparams{};
    hipGraphNode_t memcpyNodeC{};

    memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
    myparams.srcPos = make_hipPos(0, 0, 0);
    myparams.dstPos = make_hipPos(0, 0, 0);
    myparams.extent = make_hipExtent(Nbytes, 1, 1);
    myparams.srcPtr = make_hipPitchedPtr(C_h.host_ptr(), Nbytes, N, 1);
    myparams.dstPtr = make_hipPitchedPtr(C_d.ptr(), Nbytes, N, 1);
    myparams.kind = hipMemcpyHostToDevice;

    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNodeC, capInfoGraph, nullptr, 0, &myparams));

    // Replace capture dependency with new memcpy node created.
    // Further nodes captured in stream will depend on the new memcpy node.
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyNodeC, 1,
                                                 hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capInfoGraph, &nodelist,
                                         &numDependencies));

    // Verify updating dependency is taking effect.
    REQUIRE(numDependencies == 1);
    REQUIRE(nodelist[0] == memcpyNodeC);

    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                       C_d.ptr(), C_d.ptr(), N);
  }

  SECTION("Set dependency to Kernel node depending on graph branch") {
    hipGraphNode_t kernelNode{};
    hipKernelNodeParams kernelNodeParams{};

    // Add node to modify vector sqr result and plug-in the nod
    float* C_ptr = C_d.ptr();
    float* A_ptr = A_d.ptr();
    size_t NElem{N};
    void* kernelArgs[] = {&A_ptr, &C_ptr, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<float>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode, capInfoGraph, &nodelist[0], 1, &kernelNodeParams));

    // Replace capture dependency with new kernel node created.
    // Further nodes captured in stream will depend on the new kernel node.
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &kernelNode, 1,
                                                 hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capInfoGraph, &nodelist,
                                         &numDependencies));

    // Verify updating dependency is taking effect.
    REQUIRE(numDependencies == 1);
    REQUIRE(nodelist[0] == kernelNode);
  }

  HIP_CHECK(hipMemcpyAsync(B_h.ptr(), C_d.ptr(), Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), N, static_cast<float>(i));
    std::fill_n(C_h.host_ptr(), N, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i) * static_cast<float>(i), N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Local Function for adding new dependency
 */
static void UpdateStreamCaptureDependenciesAdd(hipStream_t stream,
                                               hipStreamCaptureMode captureMode) {
  constexpr size_t N = 1000000;
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(float);

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  hipGraph_t capInfoGraph{nullptr};
  const hipGraphNode_t* nodelist{};
  size_t numDependencies;

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> C_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> B_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> C_d(LinearAllocs::hipMalloc, Nbytes);

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  EventsGuard events_guard(3);
  StreamsGuard streams_guard(2);

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceBranched(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream,
                          streams_guard.stream_list(), events_guard.event_list());

  constexpr int numDepsCreated = 2;  // Num of dependencies created

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capInfoGraph, &nodelist,
                                       &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capInfoGraph != nullptr);
  REQUIRE(numDependencies == numDepsCreated);

  SECTION("Add Dependency to independant Memcpy node") {
    // Create memcpy node and add it as additional dependency in graph
    hipMemcpy3DParms myparams{};
    hipGraphNode_t memcpyNodeC{};

    memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
    myparams.srcPos = make_hipPos(0, 0, 0);
    myparams.dstPos = make_hipPos(0, 0, 0);
    myparams.extent = make_hipExtent(Nbytes, 1, 1);
    myparams.srcPtr = make_hipPitchedPtr(C_h.host_ptr(), Nbytes, N, 1);
    myparams.dstPtr = make_hipPitchedPtr(C_d.ptr(), Nbytes, N, 1);
    myparams.kind = hipMemcpyHostToDevice;

    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNodeC, capInfoGraph, nullptr, 0, &myparams));

    // Add/Append additional dependency MemcpyNodeC to the existing set.
    // Further nodes captured in stream will depend on Memcpy nodes A, B and C.
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyNodeC, 1,
                                                 hipStreamAddCaptureDependencies));
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capInfoGraph, &nodelist,
                                         &numDependencies));

    REQUIRE(numDependencies == numDepsCreated + 1);

    hipLaunchKernelGGL(vectorSum, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d.ptr(),
                       C_d.ptr(), B_d.ptr(), N);
  }

  SECTION("Add Dependency to Kernel node depending on graph branch") {
    hipGraphNode_t kernelNode{};
    hipKernelNodeParams kernelNodeParams{};

    // Add node to modify vector sqr result and plug-in the nod
    float* C_ptr = C_d.ptr();
    float* A_ptr = A_d.ptr();
    size_t NElem{N};
    void* kernelArgs[] = {&A_ptr, &C_ptr, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(vectorSet);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode, capInfoGraph, &nodelist[0], 1, &kernelNodeParams));

    // Add/Append additional dependency addNode to the existing set.
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &kernelNode, 1,
                                                 hipStreamAddCaptureDependencies));

    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capInfoGraph, &nodelist,
                                         &numDependencies));

    REQUIRE(numDependencies == numDepsCreated + 1);

    hipLaunchKernelGGL(vectorSum, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d.ptr(),
                       C_d.ptr(), B_d.ptr(), N);
  }

  HIP_CHECK(hipMemcpyAsync(B_h.ptr(), B_d.ptr(), Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), N, static_cast<float>(i));
    std::fill_n(C_h.host_ptr(), N, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i) * 2, N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify replacing existing dependency set with new nodes by
 * calling the api with flag hipStreamSetCaptureDependencies for
 * created/hipStreamPerThread for all capture modes. Verify updated dependency
 * list is taking effect:
 *        -# Replace existing dependencies with a new memcpy node that has no
 * dependencies
 *        -# Replace existing dependencies with a new kernel node which depends
 * on a previously captured sequence
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamUpdateCaptureDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipStreamSetCaptureDependencies_Positive_Functional") {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  UpdateStreamCaptureDependenciesSet(stream, captureMode);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify adding additional depencies in the flow by calling the
 * api with flag hipStreamAddCaptureDependencies for created/hipStreamPerThread
 * for all capture modes. Verify updated dependency list is taking effect:
 *        -# Add new memcpy node that has no parent to the existing dependecies
 *        -# Add new kernel node which depends on a previously captured sequence
 * to the existing dependencies
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamUpdateCaptureDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipStreamAddCaptureDependencies_Positive_Functional") {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  UpdateStreamCaptureDependenciesAdd(stream, captureMode);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify when dependencies are passed as nullptr and numDeps as 0,
 * api returns success
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamUpdateCaptureDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipStreamUpdateCaptureDependencies_Positive_Parameters") {
  hipGraph_t graph{nullptr};

  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
  const hipStreamUpdateCaptureDependenciesFlags flag =
      GENERATE(hipStreamAddCaptureDependencies, hipStreamSetCaptureDependencies);

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));  // hipStreamCaptureModeGlobal));

  HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, nullptr, 0, flag));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Pass Dependencies as nullptr and numDeps as nonzero
 *        -# numDeps exceeds actual number of nodes
 *        -# Invalid flag is passed
 *        -# Dependency node is a un-initialized/invalid parameter
 *        -# Stream is not capturing
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamUpdateCaptureDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipStreamUpdateCaptureDependencies_Negative_Parameters") {
  const int Nbytes = 100;
  hipGraph_t capInfoGraph{nullptr};
  hipGraph_t graph{nullptr};

  hipStreamCaptureStatus captureStatus;
  size_t numDependencies;
  const hipGraphNode_t* nodelist{};
  hipGraphNode_t memsetNode{};
  std::vector<hipGraphNode_t> dependencies;

  LinearAllocGuard<char> A_d(LinearAllocs::hipMalloc, Nbytes);

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(A_d.ptr(), 0, Nbytes, stream));

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capInfoGraph, &nodelist,
                                       &numDependencies));

  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(A_d.ptr());
  memsetParams.value = 1;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(
      hipGraphAddMemsetNode(&memsetNode, capInfoGraph, nodelist, numDependencies, &memsetParams));
  dependencies.push_back(memsetNode);

  SECTION("Dependencies as nullptr and numDeps as nonzero") {
    HIP_CHECK_ERROR(hipStreamUpdateCaptureDependencies(stream, nullptr, dependencies.size(),
                                                       hipStreamAddCaptureDependencies),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    constexpr int invalidFlag = 20;
    HIP_CHECK_ERROR(hipStreamUpdateCaptureDependencies(stream, dependencies.data(),
                                                       dependencies.size(), invalidFlag),
                    hipErrorInvalidValue);
  }

#if HT_NVIDIA  // EXSWHTEC-227
  SECTION("numDeps exceeding actual number of nodes") {
    HIP_CHECK_ERROR(
        hipStreamUpdateCaptureDependencies(stream, dependencies.data(), dependencies.size() + 1,
                                           hipStreamAddCaptureDependencies),
        hipErrorInvalidValue);
  }

  SECTION("depnode as un-initialized/invalid parameter") {
    hipGraphNode_t uninit_node{};
    HIP_CHECK_ERROR(hipStreamUpdateCaptureDependencies(stream, &uninit_node, 1,
                                                       hipStreamAddCaptureDependencies),
                    hipErrorInvalidValue);
  }
#endif

#if HT_AMD  // EXSWHTEC-227
  HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, dependencies.data(), dependencies.size(),
                                               hipStreamAddCaptureDependencies));
#endif

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  SECTION("Stream is not capturing") {
    HIP_CHECK_ERROR(
        hipStreamUpdateCaptureDependencies(stream, dependencies.data(), dependencies.size(),
                                           hipStreamAddCaptureDependencies),
        hipErrorIllegalState);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
