/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#pragma clang diagnostic ignored "-Wunused-parameter"
#define SIZE (1024 * 1024)
static size_t Nbytes = SIZE * sizeof(int);

__device__ int globalOut[SIZE];

/**
 * @addtogroup hipStreamBeginCapture hipStreamBeginCapture
 * @{
 * @ingroup GraphTest
 * `hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode)` -
 * Returns the last error from a runtime call.
 */

static void verifyArrayMemset(int* A_h, int val) {
  int expected_val = val | (val << 8) | (val << 16) | (val << 24);
  for (size_t i = 0; i < SIZE; i++) {
    if (A_h[i] != expected_val) {
      INFO("Memset Validation failed at i " << i << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }
}

__device__ __host__ static void callbackFunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < SIZE; i++) {
    A[i] = i + i % 2;
  }
}

__global__ static void kCallbackFunc(void* A_h) { callbackFunc(A_h); }

static void verifyCallbackFunc(int* A_h) {
  for (size_t i = 0; i < SIZE; i++) {
    if (A_h[i] != static_cast<int>(i + i % 2)) {
      INFO("CallBack Validation failed i " << i << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }
}

__global__ static void addGpuKernel(int* i_d) { *i_d = *i_d + 1; }

static void CpuCallback(void* args) {
  // do nothing function
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipStreamBeginCapture, hipStreamEndCapture status with
 *    hipGraphAddHostNode api call.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddNodeBeginCapture.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamBeginCapture_with_hipGraphAddHostNode") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t cpuGraphNode;
  int* i_d;
  HIP_CHECK(hipMalloc(&i_d, sizeof(int)));
  REQUIRE(i_d != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipHostNodeParams p = {0, 0};
  p.fn = CpuCallback;
  p.userData = nullptr;
  HIP_CHECK(hipGraphAddHostNode(&cpuGraphNode, graph, nullptr, 0, &p));
  HIP_CHECK(hipGraphDestroy(graph));
  addGpuKernel<<<1, 1, 0, stream>>>(i_d);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipFree(i_d));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Capture graph sequence using hipStreamBeginCapture and try to add a new
 *    node to the capture stream using hipStreamUpdateCaptureDependencies api
 *    which will copy back the result from the existing graph and verify
 *  1) Add a hipGraphAddMemcpyNode1D node before hipStreamEndCapture
 *  2) Add a hipGraphAddMemsetNode node before hipStreamEndCapture
 *  3) Add a hipGraphAddMemcpyNode node before hipStreamEndCapture
 *  4) Add a hipGraphAddKernelNode node before hipStreamEndCapture
 *  5) Add a hipGraphAddMemcpyNodeToSymbol and hipGraphAddMemcpyNodeFromSymbol
 *     node before hipStreamEndCapture
 *  6) Add a hipGraphAddHostNode node before hipStreamEndCapture
 *  7) Add a hipGraphAddChildGraphNode node before hipStreamEndCapture
 *  8) Add a hipGraphAddEmptyNode node before hipStreamEndCapture
 *  9) Add a hipGraphAddEventRecordNode node before hipStreamEndCapture
 *  10) Add a hipGraphAddEventWaitNode node before hipStreamEndCapture
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddNodeBeginCapture.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamEndCapture_later_and_add_a_node_inbetween") {
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyD2H_C;
  hipStream_t stream;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, SIZE, false);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  HipTest::vectorADD<int><<<1, 1, 0, stream>>>(A_d, B_d, C_d, SIZE);

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  hipGraph_t capGraph{nullptr};
  const hipGraphNode_t* nodelist{};
  size_t numDependencies;

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capGraph, &nodelist,
                                       &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capGraph != nullptr);

  SECTION("Add a hipGraphAddMemcpyNode1D node before hipStreamEndCapture") {
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, capGraph, nodelist, numDependencies, C_h, C_d,
                                      Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyD2H_C, 1,
                                                 hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add a hipGraphAddMemsetNode node before hipStreamEndCapture") {
    hipGraphNode_t memsetNode;
    int memSetVal = 7;
    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(C_d);
    memsetParams.value = memSetVal;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(
        hipGraphAddMemsetNode(&memsetNode, capGraph, nodelist, numDependencies, &memsetParams));

    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memsetNode, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyArrayMemset(C_h, memSetVal);
  }
  SECTION("Add a hipGraphAddMemcpyNode node before hipStreamEndCapture") {
    hipMemcpy3DParms myparams;
    hipGraphNode_t memcpyNode;

    memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
    myparams.srcPos = make_hipPos(0, 0, 0);
    myparams.dstPos = make_hipPos(0, 0, 0);
    myparams.srcPtr = make_hipPitchedPtr(C_d, Nbytes, 1, 1);
    myparams.dstPtr = make_hipPitchedPtr(C_h, Nbytes, 1, 1);
    myparams.extent = make_hipExtent(Nbytes, 1, 1);
    myparams.kind = hipMemcpyDeviceToHost;

    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, capGraph, nodelist, numDependencies, &myparams));

    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyNode, 1,
                                                 hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add a hipGraphAddKernelNode node before hipStreamEndCapture") {
    hipGraphNode_t kNode;
    hipKernelNodeParams kNodeParams{};
    memset(&kNodeParams, 0x00, sizeof(kNodeParams));
    void* kernelArgs[] = {&C_d};
    kNodeParams.func = reinterpret_cast<void*>(kCallbackFunc);
    kNodeParams.gridDim = dim3(1);
    kNodeParams.blockDim = dim3(256);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode, capGraph, nodelist, numDependencies, &kNodeParams));

    HIP_CHECK(
        hipStreamUpdateCaptureDependencies(stream, &kNode, 1, hipStreamSetCaptureDependencies));
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyCallbackFunc(C_h);
  }
  SECTION("Add hipGraphAddMemcpyNodeToSymbol node before hipStreamEndCapture") {
    hipGraphNode_t memcpyToSymNode, memcpyFromSymNode;

    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymNode, capGraph, nodelist, numDependencies,
                                            HIP_SYMBOL(globalOut), C_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymNode, capGraph, nullptr, 0, C_h,
                                              HIP_SYMBOL(globalOut), Nbytes, 0,
                                              hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(capGraph, &memcpyToSymNode, &memcpyFromSymNode, 1));

    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyToSymNode, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyFromSymNode, 1,
                                                 hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add a hipGraphAddHostNode node before hipStreamEndCapture") {
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, capGraph, nodelist, numDependencies, C_h, C_d,
                                      Nbytes, hipMemcpyDeviceToHost));

    hipGraphNode_t hostNode;
    hipHostNodeParams hostParams = {0, 0};
    hostParams.fn = callbackFunc;
    hostParams.userData = C_h;
    HIP_CHECK(hipGraphAddHostNode(&hostNode, capGraph, nullptr, 0, &hostParams));

    HIP_CHECK(hipGraphAddDependencies(capGraph, &memcpyD2H_C, &hostNode, 1));

    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyD2H_C, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(
        hipStreamUpdateCaptureDependencies(stream, &hostNode, 1, hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyCallbackFunc(C_h);
  }
  SECTION("Add a hipGraphAddChildGraphNode node before hipStreamEndCapture") {
    hipGraph_t childGraph;
    hipGraphNode_t memsetNode, childGraphNode;
    int memSetVal = 7;

    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(C_d);
    memsetParams.value = memSetVal;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;

    HIP_CHECK(hipGraphCreate(&childGraph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, childGraph, nullptr, 0, &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, childGraph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphAddDependencies(childGraph, &memsetNode, &memcpyD2H_C, 1));

    HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, capGraph, nodelist, numDependencies,
                                        childGraph));
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &childGraphNode, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyArrayMemset(C_h, memSetVal);
    HIP_CHECK(hipGraphDestroy(childGraph));
  }
  SECTION("Add a hipGraphAddEmptyNode node before hipStreamEndCapture") {
    hipGraphNode_t emptyNode;
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, capGraph, nodelist, numDependencies));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, capGraph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(capGraph, &emptyNode, &memcpyD2H_C, 1));

    HIP_CHECK(
        hipStreamUpdateCaptureDependencies(stream, &emptyNode, 1, hipStreamSetCaptureDependencies));
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyD2H_C, 1,
                                                 hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add hipGraphAddEventRecordNode node before hipStreamEndCapture") {
    hipGraphNode_t event_start, event_end;
    hipEvent_t eventstart, eventend;

    HIP_CHECK(hipEventCreate(&eventstart));
    HIP_CHECK(hipEventCreate(&eventend));

    HIP_CHECK(
        hipGraphAddEventRecordNode(&event_start, capGraph, nodelist, numDependencies, eventstart));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, capGraph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddEventRecordNode(&event_end, capGraph, nullptr, 0, eventend));

    HIP_CHECK(hipGraphAddDependencies(capGraph, &event_start, &memcpyD2H_C, 1));
    HIP_CHECK(hipGraphAddDependencies(capGraph, &memcpyD2H_C, &event_end, 1));

    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &event_start, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyD2H_C, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(
        hipStreamUpdateCaptureDependencies(stream, &event_end, 1, hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventSynchronize(eventend));

    // Verify execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, SIZE);

    float t = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&t, eventstart, eventend));
    REQUIRE(t > 0.0f);

    HIP_CHECK(hipEventDestroy(eventstart));
    HIP_CHECK(hipEventDestroy(eventend));
  }
  SECTION("Add hipGraphAddEventWaitNode node before hipStreamEndCapture") {
    hipGraphNode_t eventRecNode, eventWaitNode;
    hipEvent_t event;

    HIP_CHECK(hipEventCreate(&event));

    HIP_CHECK(
        hipGraphAddEventRecordNode(&eventRecNode, capGraph, nodelist, numDependencies, event));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, capGraph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddEventWaitNode(&eventWaitNode, capGraph, nullptr, 0, event));

    HIP_CHECK(hipGraphAddDependencies(capGraph, &eventRecNode, &memcpyD2H_C, 1));
    HIP_CHECK(hipGraphAddDependencies(capGraph, &memcpyD2H_C, &eventWaitNode, 1));

    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &eventRecNode, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &memcpyD2H_C, 1,
                                                 hipStreamSetCaptureDependencies));
    HIP_CHECK(hipStreamUpdateCaptureDependencies(stream, &eventWaitNode, 1,
                                                 hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventSynchronize(event));

    // Verify execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, SIZE);

    HIP_CHECK(hipEventDestroy(event));
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(capGraph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Capture graph sequence using hipStreamBeginCapture and hipStreamEndCapture
 *    Try to add a new node and link this new node to the existing graph
 *    which will copy back the result from the existing graph and verify
 *  1) Add a hipGraphAddMemcpyNode1D node after hipStreamEndCapture
 *  2) Add a hipGraphAddMemsetNode node after hipStreamEndCapture
 *  3) Add a hipGraphAddMemcpyNode node after hipStreamEndCapture
 *  4) Add a hipGraphAddKernelNode node after hipStreamEndCapture
 *  5) Add a hipGraphAddMemcpyNodeToSymbol and hipGraphAddMemcpyNodeFromSymbol
 *     node after hipStreamEndCapture
 *  6) Add a hipGraphAddHostNode node after hipStreamEndCapture
 *  7) Add a hipGraphAddChildGraphNode node after hipStreamEndCapture
 *  8) Add a hipGraphAddEmptyNode node after hipStreamEndCapture
 *  9) Add a hipGraphAddEventRecordNode node after hipStreamEndCapture
 *  10) Add a hipGraphAddEventWaitNode node after hipStreamEndCapture
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddNodeBeginCapture.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamEndCapture_first_and_add_a_node_later") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyD2H_C;
  hipStream_t stream;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, SIZE, false);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  HipTest::vectorSUB<int><<<1, 1, 0, stream>>>(A_d, B_d, C_d, SIZE);
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  size_t numN{};
  int foundAt = -1;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numN));

  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t*>(malloc(numN * sizeof(hipGraphNode_t)));
  REQUIRE(nodes != nullptr);

  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numN));
  hipGraphNodeType nodeType;
  for (int i = 0; i < numN; i++) {
    HIP_CHECK(hipGraphNodeGetType(nodes[i], &nodeType));
    if (nodeType == hipGraphNodeTypeKernel) {
      foundAt = i;
      break;
    }
  }

  SECTION("Add a hipGraphAddMemcpyNode1D node after hipStreamEndCapture") {
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &memcpyD2H_C, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorSUB(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add a hipGraphAddMemsetNode node after hipStreamEndCapture") {
    hipGraphNode_t memsetNode;
    int memSetVal = 7;
    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(C_d);
    memsetParams.value = memSetVal;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &memsetNode, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memsetNode, &memcpyD2H_C, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyArrayMemset(C_h, memSetVal);
  }
  SECTION("Add a hipGraphAddMemcpyNode node after hipStreamEndCapture") {
    hipMemcpy3DParms myparams;
    hipGraphNode_t memcpyNode;

    memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
    myparams.srcPos = make_hipPos(0, 0, 0);
    myparams.dstPos = make_hipPos(0, 0, 0);
    myparams.srcPtr = make_hipPitchedPtr(C_d, Nbytes, 1, 1);
    myparams.dstPtr = make_hipPitchedPtr(C_h, Nbytes, 1, 1);
    myparams.extent = make_hipExtent(Nbytes, 1, 1);
    myparams.kind = hipMemcpyDeviceToHost;

    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &myparams));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &memcpyNode, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorSUB(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add a hipGraphAddKernelNode node after hipStreamEndCapture") {
    hipGraphNode_t kNode;
    hipKernelNodeParams kNodeParams{};
    memset(&kNodeParams, 0x00, sizeof(kNodeParams));
    void* kernelArgs[] = {&C_d};
    kNodeParams.func = reinterpret_cast<void*>(kCallbackFunc);
    kNodeParams.gridDim = dim3(1);
    kNodeParams.blockDim = dim3(256);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &kNode, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kNode, &memcpyD2H_C, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyCallbackFunc(C_h);
  }
  SECTION("Add hipGraphAddMemcpyNodeToSymbol node after hipStreamEndCapture") {
    hipGraphNode_t memcpyToSymNode, memcpyFromSymNode;
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymNode, graph, nullptr, 0,
                                            HIP_SYMBOL(globalOut), C_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymNode, graph, nullptr, 0, C_h,
                                              HIP_SYMBOL(globalOut), Nbytes, 0,
                                              hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &memcpyToSymNode, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyToSymNode, &memcpyFromSymNode, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorSUB(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add hipGraphAddHostNode node after hipStreamEndCapture") {
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    hipGraphNode_t hostNode;
    hipHostNodeParams hostParams = {0, 0};
    hostParams.fn = callbackFunc;
    hostParams.userData = C_h;
    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &memcpyD2H_C, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_C, &hostNode, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyCallbackFunc(C_h);
  }
  SECTION("Add hipGraphAddChildGraphNode node after hipStreamEndCapture") {
    hipGraph_t childGraph;
    hipGraphNode_t memsetNode, childGraphNode;
    int memSetVal = 7;

    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(C_d);
    memsetParams.value = memSetVal;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;

    HIP_CHECK(hipGraphCreate(&childGraph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, childGraph, nullptr, 0, &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, childGraph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphAddDependencies(childGraph, &memsetNode, &memcpyD2H_C, 1));

    HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &childGraphNode, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    verifyArrayMemset(C_h, memSetVal);
    HIP_CHECK(hipGraphDestroy(childGraph));
  }
  SECTION("Add hipGraphAddEmptyNode node after hipStreamEndCapture") {
    hipGraphNode_t emptyNode;
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &emptyNode, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode, &memcpyD2H_C, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify execution result
    HipTest::checkVectorSUB(A_h, B_h, C_h, SIZE);
  }
  SECTION("Add hipGraphAddEventRecordNode node after hipStreamEndCapture") {
    hipGraphNode_t event_start, event_end;
    hipEvent_t eventstart, eventend;

    HIP_CHECK(hipEventCreate(&eventstart));
    HIP_CHECK(hipEventCreate(&eventend));

    HIP_CHECK(hipGraphAddEventRecordNode(&event_start, graph, nullptr, 0, eventstart));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddEventRecordNode(&event_end, graph, nullptr, 0, eventend));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &event_start, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &event_start, &memcpyD2H_C, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_C, &event_end, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventSynchronize(eventend));

    // Verify execution result
    HipTest::checkVectorSUB(A_h, B_h, C_h, SIZE);

    float t = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&t, eventstart, eventend));
    REQUIRE(t > 0.0f);

    HIP_CHECK(hipEventDestroy(eventstart));
    HIP_CHECK(hipEventDestroy(eventend));
  }
  SECTION("Add hipGraphAddEventWaitNode node after hipStreamEndCapture") {
    hipGraphNode_t eventRecNode, eventWaitNode;
    hipEvent_t event;

    HIP_CHECK(hipEventCreate(&event));

    HIP_CHECK(hipGraphAddEventRecordNode(&eventRecNode, graph, nullptr, 0, event));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddEventWaitNode(&eventWaitNode, graph, nullptr, 0, event));

    HIP_CHECK(hipGraphAddDependencies(graph, &nodes[foundAt], &eventRecNode, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &eventRecNode, &memcpyD2H_C, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_C, &eventWaitNode, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventSynchronize(event));

    // Verify execution result
    HipTest::checkVectorSUB(A_h, B_h, C_h, SIZE);

    HIP_CHECK(hipEventDestroy(event));
  }

  free(nodes);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Capture graph sequence using hipStreamBeginCapture and hipStreamEndCapture
 *    Add some new node to the same graph and execute it and verify
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddNodeBeginCapture.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamEndCapture_first_and_add_other_graph_node_later") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_AC, memcpyH2D_C;
  hipStream_t stream;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, SIZE, false);

  int *A_d1, *B_d1, *C_d1, *A_h1, *B_h1, *C_h1;
  HipTest::initArrays(&A_d1, &B_d1, &C_d1, &A_h1, &B_h1, &C_h1, SIZE, false);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d1, A_h1, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d1, B_h1, Nbytes, hipMemcpyHostToDevice, stream));
  HipTest::vectorADD<int><<<1, 1, 0, stream>>>(A_d1, B_d1, C_d1, SIZE);
  HIP_CHECK(hipMemcpyAsync(C_h1, C_d1, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr, 0, C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr, 0, A_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackFunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_AC, &hostNode, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify execution result fir above two graph operations
  verifyCallbackFunc(A_h);
  HipTest::checkVectorADD(A_h1, B_h1, C_h1, SIZE);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HipTest::freeArrays(A_d1, B_d1, C_d1, A_h1, B_h1, C_h1, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Capture graph sequence using hipStreamBeginCapture and
 *    add some new node before hipStreamEndCapture to the same graph
 *    and hipGraphAddEmptyNode to use as last node to grah to complete.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddNodeBeginCapture.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamEndCapture_later_and_addEmptyNode") {
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyD2H_C;
  hipStream_t stream;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, SIZE, false);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  HipTest::vectorSUB<int><<<1, 1, 0, stream>>>(A_d, B_d, C_d, SIZE);

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  hipGraph_t capGraph{nullptr};
  const hipGraphNode_t* nodelist{};
  size_t numDependencies;

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, nullptr, &capGraph, &nodelist,
                                       &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capGraph != nullptr);

  hipGraphNode_t memsetNode;
  int memSetVal = 7;
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(C_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, capGraph, nodelist, numDependencies, &memsetParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, capGraph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  hipGraphNode_t emptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, capGraph, nullptr, 0));

  HIP_CHECK(hipGraphAddDependencies(capGraph, &memsetNode, &memcpyD2H_C, 1));
  HIP_CHECK(hipGraphAddDependencies(capGraph, &memcpyD2H_C, &emptyNode, 1));

  HIP_CHECK(
      hipStreamUpdateCaptureDependencies(stream, &memsetNode, 1, hipStreamSetCaptureDependencies));
  HIP_CHECK(
      hipStreamUpdateCaptureDependencies(stream, &memcpyD2H_C, 1, hipStreamSetCaptureDependencies));
  HIP_CHECK(
      hipStreamUpdateCaptureDependencies(stream, &emptyNode, 1, hipStreamSetCaptureDependencies));

  HIP_CHECK(hipStreamEndCapture(stream, &capGraph));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, capGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify execution result
  verifyArrayMemset(C_h, memSetVal);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(capGraph));
  HIP_CHECK(hipStreamDestroy(stream));
}


/**
 * End doxygen group GraphTest.
 * @}
 */
