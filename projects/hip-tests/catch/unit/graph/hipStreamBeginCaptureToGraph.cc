/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * @addtogroup hipStreamBeginCaptureToGraph hipStreamBeginCaptureToGraph
 * @{
 * @ingroup GraphTest
 * `hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
 *                                          const hipGraphNode_t* dependencies,
 *                                          const hipGraphEdgeData* dependencyData,
 *                                          size_t numDependencies, hipStreamCaptureMode mode);` -
 *  Begins graph capture on a stream to an existing graph.
 */
#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <vector>
#include <atomic>
#include <functional>
#include <cstddef>

constexpr size_t N = 1 << 20;
constexpr unsigned blocks = 256;
constexpr unsigned threadsPerBlock = 64;

static bool CaptureStreamAndLaunchGraph(int* A_d, int* B_d, int* C_d, int* A_h, int* B_h, int* C_h,
                                        hipStreamCaptureMode mode, hipStream_t& stream1,
                                        hipStream_t& stream2, hipGraph_t& graph,
                                        bool verifyStreamSync = false,
                                        std::function<bool()> verifyFunc1 = nullptr) {
  auto verifyFunc = [&]() {
    // Validate the computation
    for (size_t i = 0; i < N; i++) {
      if (C_h[i] != (A_h[i] - B_h[i])) {
        fprintf(stderr, "Error at %zu: C=%d, A=%d, B=%d\n", i, C_h[i], A_h[i], B_h[i]);
        return false;
      }
    }
    return true;
  };

  hipGraphExec_t graphExec{nullptr};
  size_t Nbytes = N * sizeof(int);
  hipEvent_t e;
  HIP_CHECK(hipEventCreate(&e));
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0, mode));
  HIP_CHECK(hipEventRecord(e, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e, 0));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream1));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipEventRecord(e, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, e, 0));
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, stream1>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  bool res = true;
  if (verifyStreamSync) {
    // Verify if hipStreamSynchronize() works as expected
    res = verifyFunc1 ? verifyFunc1() : true;
    res = res && verifyFunc();
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipEventDestroy(e));

  if (!verifyStreamSync) {
    // After hipGraphExecDestroy(), all internal streams are
    // surely synced!
    res = verifyFunc1 ? verifyFunc1() : true;
    res = res && verifyFunc();
  }
  return res;
}

/**
 * Test Description
 * ------------------------
 *    - Basic Functional Test for API capturing custom stream and replaying sequence.
 * Test exercises the API on available/possible modes.
 * Stream capture with different modes behave the same when supported/
 * safe apis are used in sequence.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_BasicFunctional") {
  int *A_d, *B_d, *C_d;
  std::vector<int> A_h(N), B_h(N), C_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream1, stream2;
  bool ret;
  hipGraph_t graph{nullptr};

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 2 * i;
    B_h[i] = 2 * i + 1;
  }

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  bool verifyStreamSync = false;

  SECTION("Capture stream and launch graph when mode is global") {
    SECTION("Verify after hipGraphExecDestroy()") { verifyStreamSync = false; }
    SECTION("Verify after hipStreamSynchronize()") { verifyStreamSync = true; }
    ret = CaptureStreamAndLaunchGraph(A_d, B_d, C_d, A_h.data(), B_h.data(), C_h.data(),
                                      hipStreamCaptureModeGlobal, stream1, stream2, graph,
                                      verifyStreamSync);
  }

  SECTION("Capture stream and launch graph when mode is local") {
    SECTION("Verify after hipGraphExecDestroy()") { verifyStreamSync = false; }
    SECTION("Verify after hipStreamSynchronize()") { verifyStreamSync = true; }
    ret = CaptureStreamAndLaunchGraph(A_d, B_d, C_d, A_h.data(), B_h.data(), C_h.data(),
                                      hipStreamCaptureModeThreadLocal, stream1, stream2, graph,
                                      verifyStreamSync);
  }

  SECTION("Capture stream and launch graph when mode is relaxed") {
    SECTION("Verify after hipGraphExecDestroy()") { verifyStreamSync = false; }
    SECTION("Verify after hipStreamSynchronize()") { verifyStreamSync = true; }
    ret = CaptureStreamAndLaunchGraph(A_d, B_d, C_d, A_h.data(), B_h.data(), C_h.data(),
                                      hipStreamCaptureModeRelaxed, stream1, stream2, graph,
                                      verifyStreamSync);
  }

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  REQUIRE(ret == true);
}

/**
 * Test Description
 * ------------------------
 *    - Create a manual graph. Once done capture a stream in the same graph
 * independently. Execute the graph and verify that both the graphs
 * execute independently.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_CaptureIndepGraph") {
  int *A1_d, *B1_d, *C1_d;
  std::vector<int> A1_h(N), B1_h(N), C1_h(N);
  int *A2_d, *B2_d, *C2_d;
  std::vector<int> A2_h(N), B2_h(N), C2_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream1, stream2;
  bool ret;
  hipGraph_t graph{nullptr};
  hipGraphNode_t memcpyNode1, memcpyNode2, memcpyNode3, kernelNode;

  // Verify Manual Graph
  auto verifyFunc = [&]() {
    // Validate the computation
    for (size_t i = 0; i < N; i++) {
      if (C2_h[i] != (A2_h[i] + B2_h[i])) {
        fprintf(stderr, "Error at %zu: C2=%d, A2=%d, B2=%d\n", i, C2_h[i], A2_h[i], B2_h[i]);
        return false;
      }
    }
    return true;
  };

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A2_h[i] = A1_h[i] = 2 * i;
    B2_h[i] = B1_h[i] = 2 * i + 1;
  }

  // Create stream and empty graph
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A1_d, Nbytes));
  HIP_CHECK(hipMalloc(&B1_d, Nbytes));
  HIP_CHECK(hipMalloc(&C1_d, Nbytes));
  HIP_CHECK(hipMalloc(&A2_d, Nbytes));
  HIP_CHECK(hipMalloc(&B2_d, Nbytes));
  HIP_CHECK(hipMalloc(&C2_d, Nbytes));

  // Create manual graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode1, graph, nullptr, 0, A2_d, A2_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode2, graph, nullptr, 0, B2_d, B2_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode3, graph, nullptr, 0, C2_h.data(), C2_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  size_t arrSize = N;
  void* kernelArgs[4] = {reinterpret_cast<void*>(&A2_d), reinterpret_cast<void*>(&B2_d),
                         reinterpret_cast<void*>(&C2_d), &arrSize};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks, 1, 1);
  kernelNodeParams.blockDim = dim3(threadsPerBlock, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, 0, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyNode1, &kernelNode, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyNode2, &kernelNode, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelNode, &memcpyNode3, 1));
  bool verifyStreamSync = false;

  // Capture an independent graph from stream
  SECTION("Verify after hipGraphExecDestroy()") { verifyStreamSync = false; }
  SECTION("Verify after hipStreamSynchronize()") { verifyStreamSync = true; }
  ret = CaptureStreamAndLaunchGraph(A1_d, B1_d, C1_d, A1_h.data(), B1_h.data(), C1_h.data(),
                                    hipStreamCaptureModeGlobal, stream1, stream2, graph,
                                    verifyStreamSync, verifyFunc);
  REQUIRE(ret == true);

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A1_d));
  HIP_CHECK(hipFree(B1_d));
  HIP_CHECK(hipFree(C1_d));
  HIP_CHECK(hipFree(A2_d));
  HIP_CHECK(hipFree(B2_d));
  HIP_CHECK(hipFree(C2_d));
  REQUIRE(ret == true);
}

/**
 * Test Description
 * ------------------------
 *    - Create a manual graph. Once done capture a dependent graph
 * from stream. Launch the graph and validate results.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
#ifdef __linux__
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_CaptureDepGraph") {
  hipGraphExec_t graphExec{nullptr};
  int *A1_d, *B1_d, *C1_d, *C2_d;
  std::vector<int> A1_h(N), B1_h(N), C1_h(N), C2_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream;
  hipGraph_t graph{nullptr};
  hipGraphNode_t memcpyNode1, memcpyNode2, memcpyNode3, kernelNode;

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A1_h[i] = 2 * i;
    B1_h[i] = 2 * i + 1;
  }

  // Create stream and empty graph
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A1_d, Nbytes));
  HIP_CHECK(hipMalloc(&B1_d, Nbytes));
  HIP_CHECK(hipMalloc(&C1_d, Nbytes));
  HIP_CHECK(hipMalloc(&C2_d, Nbytes));

  // Create manual graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode1, graph, nullptr, 0, A1_d, A1_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode2, graph, nullptr, 0, B1_d, B1_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode3, graph, nullptr, 0, C1_h.data(), C1_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  size_t arrSize = N;
  void* kernelArgs[4] = {reinterpret_cast<void*>(&A1_d), reinterpret_cast<void*>(&B1_d),
                         reinterpret_cast<void*>(&C1_d), &arrSize};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks, 1, 1);
  kernelNodeParams.blockDim = dim3(threadsPerBlock, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, 0, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyNode1, &kernelNode, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyNode2, &kernelNode, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelNode, &memcpyNode3, 1));

  // Capture an dependant graph from stream
  std::vector<hipGraphNode_t> vnode;
  vnode.push_back(memcpyNode1);
  vnode.push_back(memcpyNode2);
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream, graph, vnode.data(), nullptr, vnode.size(),
                                         hipStreamCaptureModeGlobal));
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, stream>>>(A1_d, B1_d, C2_d, N);
  HIP_CHECK(hipMemcpyAsync(C2_h.data(), C2_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // Instantiate and Launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  bool ret = true;
  // Verify Manual Graph
  for (size_t i = 0; i < N; i++) {
    if (C1_h[i] != (A1_h[i] + B1_h[i])) {
      fprintf(stderr, "Error at %zu: C1=%d, A1=%d, B1=%d\n", i, C1_h[i], A1_h[i], B1_h[i]);
      ret = false;
      break;
    }
    if (C2_h[i] != (A1_h[i] - B1_h[i])) {
      fprintf(stderr, "Error at %zu: C2=%d, A1=%d, B1=%d\n", i, C2_h[i], A1_h[i], B1_h[i]);
      ret = false;
      break;
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A1_d));
  HIP_CHECK(hipFree(B1_d));
  HIP_CHECK(hipFree(C1_d));
  HIP_CHECK(hipFree(C2_d));
  REQUIRE(ret == true);
}
#endif
/**
 * Test Description
 * ------------------------
 *    - Capture a complex graph involving multiple streams.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_ComplexGraph") {
  int *A_d, *B_d, *C_d, *D_d;
  std::vector<int> A_h(N), B_h(N), C_h(N), D_h(N);
  size_t Nbytes = N * sizeof(int);

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 2 * i;
    B_h[i] = 2 * i + 1;
  }

  // Create streams, events and empty graph
  hipStream_t stream1, stream2, stream3, stream4;
  hipEvent_t e1, e2, e3, e4;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipStreamCreate(&stream4));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));
  HIP_CHECK(hipEventCreate(&e3));
  HIP_CHECK(hipEventCreate(&e4));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));

  // Capture Nodes from multiple streams
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  // Capture all the other streams as well
  HIP_CHECK(hipEventRecord(e1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, e1, 0));
  HIP_CHECK(hipStreamWaitEvent(stream4, e1, 0));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h.data(), Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream3));
  HIP_CHECK(hipMemsetAsync(D_d, 0, Nbytes, stream4));
  HIP_CHECK(hipEventRecord(e1, stream1));
  HIP_CHECK(hipEventRecord(e2, stream2));
  HIP_CHECK(hipEventRecord(e3, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
  HIP_CHECK(hipStreamWaitEvent(stream1, e3, 0));
  HipTest::vectorADD<<<dim3(blocks), dim3(threadsPerBlock), 0, stream1>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipStreamWaitEvent(stream4, e2, 0));
  HIP_CHECK(hipStreamWaitEvent(stream4, e1, 0));
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, stream4>>>(A_d, B_d, D_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h.data(), C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipMemcpyAsync(D_h.data(), D_d, Nbytes, hipMemcpyDeviceToHost, stream4));
  HIP_CHECK(hipEventRecord(e4, stream4));
  HIP_CHECK(hipStreamWaitEvent(stream1, e4, 0));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  // Instantiate Graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Launch Graph
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));

  // Verify Manual Graph
  for (size_t i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (A_h[i] + B_h[i]));
    REQUIRE(D_h[i] == (A_h[i] - B_h[i]));
  }

  // Destroy all resources
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream4));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipEventDestroy(e3));
  HIP_CHECK(hipEventDestroy(e4));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
}

/**
 * Test Description
 * ------------------------
 *    - Capture a graph twice - first time using hipStreamBeginCapture/
 * hipStreamBeginCaptureToGraph followed by hipStreamBeginCaptureToGraph.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_CaptureTwice") {
  bool useSameAPI = GENERATE(true, false);
  int *A_d, *B_d, *C_d, *D_d;
  std::vector<int> A_h(N), B_h(N), C_h(N), D_h(N);
  size_t Nbytes = N * sizeof(int);

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 2 * i;
    B_h[i] = 2 * i + 1;
  }

  // Create streams, events and empty graph
  hipStream_t stream1, stream2;
  hipEvent_t e1, e2;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  if (useSameAPI) {
    HIP_CHECK(hipGraphCreate(&graph, 0));
  }
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));

  // Capture Nodes from multiple streams
  if (useSameAPI) {
    HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0,
                                           hipStreamCaptureModeGlobal));
  } else {
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  }
  // Capture all the other streams as well
  HIP_CHECK(hipEventRecord(e1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h.data(), Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipEventRecord(e2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
  HipTest::vectorADD<<<dim3(blocks), dim3(threadsPerBlock), 0, stream1>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h.data(), C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  // Get the dependant nodes
  size_t numRootNodes = 0;
  HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
  std::vector<hipGraphNode_t> rootNodes(numRootNodes);
  HIP_CHECK(hipGraphGetRootNodes(graph, rootNodes.data(), &numRootNodes));
  // Capture Nodes from multiple streams
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, rootNodes.data(), nullptr,
                                         rootNodes.size(), hipStreamCaptureModeGlobal));
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, stream1>>>(A_d, B_d, D_d, N);
  HIP_CHECK(hipMemcpyAsync(D_h.data(), D_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  // Instantiate Graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Launch Graph
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));

  // Verify Manual Graph
  for (size_t i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (A_h[i] + B_h[i]));
    REQUIRE(D_h[i] == (A_h[i] - B_h[i]));
  }

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
}

/**
 * Test Description
 * ------------------------
 *    - Capture a graph using hipStreamBeginCaptureToGraph. Clone the graph followed
 * by modifying the cloned graph using hipStreamBeginCaptureToGraph.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_ModifyCloneGraph") {
  int *A_d, *B_d, *C_d, *D_d;
  std::vector<int> A_h(N), B_h(N), C_h(N), D_h(N);
  size_t Nbytes = N * sizeof(int);

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 2 * i;
    B_h[i] = 2 * i + 1;
  }

  // Create streams, events and empty graph
  hipStream_t stream1, stream2;
  hipEvent_t e1, e2;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));

  // Capture Nodes from multiple streams
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  // Capture all the other streams as well
  HIP_CHECK(hipEventRecord(e1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h.data(), Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipEventRecord(e2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
  HipTest::vectorADD<<<dim3(blocks), dim3(threadsPerBlock), 0, stream1>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h.data(), C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  // Clone the graph
  hipGraph_t graphClone{nullptr};
  HIP_CHECK(hipGraphClone(&graphClone, graph));
  // Get the dependant nodes
  size_t numRootNodes = 0;
  HIP_CHECK(hipGraphGetRootNodes(graphClone, nullptr, &numRootNodes));
  std::vector<hipGraphNode_t> rootNodes(numRootNodes);
  HIP_CHECK(hipGraphGetRootNodes(graphClone, rootNodes.data(), &numRootNodes));
  // Capture Nodes from multiple streams
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graphClone, rootNodes.data(), nullptr,
                                         rootNodes.size(), hipStreamCaptureModeGlobal));
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, stream1>>>(A_d, B_d, D_d, N);
  HIP_CHECK(hipMemcpyAsync(D_h.data(), D_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graphClone));

  // Instantiate Cloned Graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graphClone, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Launch Cloned Graph
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));

  // Verify Manual Graph
  for (size_t i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (A_h[i] + B_h[i]));
    REQUIRE(D_h[i] == (A_h[i] - B_h[i]));
  }

  // Destroy all resources
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graphClone));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
}

/**
 * Test Description
 * ------------------------
 *    - Create a manual graph with child graph node. Capture this graph
 * using hipStreamBeginCaptureToGraph and add new nodes.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_CaptureChildpGraph") {
  int *A_d, *B_d, *C_d, *D_d;
  std::vector<int> A_h(N), B_h(N), C_h(N), D_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream;
  hipGraph_t graph{nullptr}, graphChild{nullptr};
  hipGraphNode_t memcpyNode1, memcpyNode2, memcpyNode3, kernelNode1;

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
    B_h[i] = N - i - 1;
  }

  // Create stream and empty graph
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&graphChild, 0));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));

  // Create manual graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode1, graph, nullptr, 0, A_d, A_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode2, graph, nullptr, 0, B_d, B_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode3, graphChild, nullptr, 0, C_h.data(), C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  size_t arrSize = N;
  void* kernelArgs[4] = {reinterpret_cast<void*>(&A_d), reinterpret_cast<void*>(&B_d),
                         reinterpret_cast<void*>(&C_d), &arrSize};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks, 1, 1);
  kernelNodeParams.blockDim = dim3(threadsPerBlock, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  // Create the child graph node kernelNode1->memcpyNode3
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode1, graphChild, 0, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graphChild, &kernelNode1, &memcpyNode3, 1));
  // Add the graphChild to graph with dependencies
  std::vector<hipGraphNode_t> dependncy;
  dependncy.push_back(memcpyNode1);
  dependncy.push_back(memcpyNode2);
  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, dependncy.data(), dependncy.size(),
                                      graphChild));
  // Capture stream into graph
  // Capture Nodes from multiple streams
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream, graph, dependncy.data(), nullptr, dependncy.size(),
                                         hipStreamCaptureModeGlobal));
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, stream>>>(A_d, B_d, D_d, N);
  HIP_CHECK(hipMemcpyAsync(D_h.data(), D_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  // Instantiate Cloned Graph
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Launch Cloned Graph
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  // Verify Manual Graph
  for (size_t i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (A_h[i] + B_h[i]));
    REQUIRE(D_h[i] == (A_h[i] - B_h[i]));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graphChild));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
}

/**
 * Test Description
 * ------------------------
 *    - Create a manual graph with child graph node. Capture the child graph
 * using hipStreamBeginCaptureToGraph and add new nodes.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_ModifyChildpGraph") {
  int *A_d, *B_d, *C_d, *D_d;
  std::vector<int> A_h(N), B_h(N), C_h(N), D_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream;
  hipGraph_t graph{nullptr}, graphChild{nullptr};
  hipGraphNode_t memcpyNode1, memcpyNode2, memcpyNode3, kernelNode1;

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
    B_h[i] = N - i - 1;
  }

  // Create stream and empty graph
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&graphChild, 0));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));

  // Create manual graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode1, graph, nullptr, 0, A_d, A_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode2, graph, nullptr, 0, B_d, B_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode3, graphChild, nullptr, 0, C_h.data(), C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  size_t arrSize = N;
  void* kernelArgs[4] = {reinterpret_cast<void*>(&A_d), reinterpret_cast<void*>(&B_d),
                         reinterpret_cast<void*>(&C_d), &arrSize};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks, 1, 1);
  kernelNodeParams.blockDim = dim3(threadsPerBlock, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  // Create the child graph node kernelNode1->memcpyNode3
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode1, graphChild, 0, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graphChild, &kernelNode1, &memcpyNode3, 1));
  // Add the graphChild to graph with dependencies
  std::vector<hipGraphNode_t> dependncy;
  dependncy.push_back(memcpyNode1);
  dependncy.push_back(memcpyNode2);
  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, dependncy.data(), dependncy.size(),
                                      graphChild));
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode, &graphChild));
  // Capture stream into graph
  // Capture Nodes from multiple streams
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream, graphChild, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, stream>>>(A_d, B_d, D_d, N);
  HIP_CHECK(hipMemcpyAsync(D_h.data(), D_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graphChild));
  // Instantiate Cloned Graph
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Launch Cloned Graph
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  // Verify Manual Graph
  for (size_t i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (A_h[i] + B_h[i]));
    REQUIRE(D_h[i] == (A_h[i] - B_h[i]));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_Negative") {
  // Create streams and graph
  hipStream_t stream;
  hipGraph_t graph{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  SECTION("Null graph") {
    REQUIRE(hipErrorInvalidValue == hipStreamBeginCaptureToGraph(stream, nullptr, nullptr, nullptr,
                                                                 0, hipStreamCaptureModeGlobal));
  }
  SECTION("Null dependencies") {
    REQUIRE(hipErrorInvalidValue == hipStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 1,
                                                                 hipStreamCaptureModeGlobal));
  }
  SECTION("Invalid mode") {
    REQUIRE(hipErrorInvalidValue ==
            hipStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                         static_cast<hipStreamCaptureMode>(-1)));
  }
  SECTION("Capturing a stream already in Capture mode") {
    hipGraph_t graphLoc{nullptr};
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    hipError_t err = hipStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                                  hipStreamCaptureModeGlobal);
    REQUIRE(hipErrorIllegalState == err);
    HIP_CHECK(hipStreamEndCapture(stream, &graphLoc));
    HIP_CHECK(hipGraphDestroy(graphLoc));
  }
  // Destroy all resources
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Capturing empty graph and testing state of stream.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_StateTesting") {
  // Create streams and graph
  hipStream_t stream1, stream2;
  hipEvent_t e;
  hipGraph_t graph{nullptr};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&e));
  hipStreamCaptureStatus captureStatus = hipStreamCaptureStatusNone;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamIsCapturing(stream1, &captureStatus));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamIsCapturing(stream1, &captureStatus));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  HIP_CHECK(hipEventRecord(e, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e, 0));
  HIP_CHECK(hipStreamIsCapturing(stream2, &captureStatus));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  HIP_CHECK(hipEventRecord(e, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, e, 0));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  HIP_CHECK(hipStreamIsCapturing(stream1, &captureStatus));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  // Destroy all resources
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(e));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Capturing empty graph and validate Capture Info.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_GetCaptureInfo") {
  // Create streams and graph
  hipStream_t stream1, stream2;
  hipEvent_t e;
  hipGraph_t graph{nullptr};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&e));
  hipStreamCaptureStatus captureStatus = hipStreamCaptureStatusNone;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  unsigned long long id_before = 0, id_after = 0, id_strm1 = 0, id_strm2 = 0;  // NOLINT
  HIP_CHECK(hipStreamGetCaptureInfo(stream1, &captureStatus, &id_before));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamGetCaptureInfo(stream1, &captureStatus, &id_strm1));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  HIP_CHECK(hipEventRecord(e, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e, 0));
  HIP_CHECK(hipStreamGetCaptureInfo(stream2, &captureStatus, &id_strm2));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  HIP_CHECK(hipEventRecord(e, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, e, 0));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  HIP_CHECK(hipStreamGetCaptureInfo(stream1, &captureStatus, &id_after));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  REQUIRE(id_strm1 == id_strm2);
  // Destroy all resources
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(e));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Abruptly end stream capture when either stream capture
 * is still in progress or stream has operations queued which are not yet
 * captured.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamBeginCaptureToGraph_EndingWhileCaptureInProgress") {
  hipStream_t stream1, stream2;
  hipGraph_t graph{nullptr};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  size_t Nbytes = N * sizeof(int);
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  int *A_d, *B_d, *C_d;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  std::vector<int> A_h(N);
  SECTION("Abruptly end strm capture when in progress in forked strm") {
    hipEvent_t e;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0,
                                           hipStreamCaptureModeGlobal));
    HIP_CHECK(hipEventRecord(e, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, e, 0));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), Nbytes, hipMemcpyHostToDevice, stream1));
    REQUIRE(hipSuccess == hipStreamEndCapture(stream1, &graph));
    HIP_CHECK(hipEventDestroy(e));
  }

  SECTION("End strm capture when forked strm still has operations") {
    hipEvent_t e1, e2;
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventCreate(&e2));
    HIP_CHECK(hipStreamBeginCaptureToGraph(stream1, graph, nullptr, nullptr, 0,
                                           hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), Nbytes, hipMemcpyHostToDevice, stream1));
    HIP_CHECK(hipEventRecord(e1, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
    HIP_CHECK(hipMemcpyAsync(B_d, A_h.data(), Nbytes, hipMemcpyHostToDevice, stream2));
    HIP_CHECK(hipEventRecord(e2, stream2));
    HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
    HIP_CHECK(hipMemcpyAsync(C_d, A_h.data(), Nbytes, hipMemcpyHostToDevice, stream2));
    REQUIRE(hipErrorStreamCaptureUnjoined == hipStreamEndCapture(stream1, &graph));
    HIP_CHECK(hipEventDestroy(e2));
    HIP_CHECK(hipEventDestroy(e1));
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}

/**
 * Test Description
 * ------------------------
 *    - Capture Graph using 2 different streams of different properties and validate the
 * captured graph.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
enum class strmFlag { defFlag, sameFlag, diffFlag, diffPrio };

TEST_CASE("Unit_hipStreamBeginCaptureToGraph_MultipleFlags") {
  strmFlag flag =
      GENERATE(strmFlag::defFlag, strmFlag::sameFlag, strmFlag::diffFlag, strmFlag::diffPrio);
  int *A_d, *B_d, *C_d;
  std::vector<int> A_h(N), B_h(N), C_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream1, stream2;
  bool ret;
  hipGraph_t graph{nullptr};

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 2 * i;
    B_h[i] = 2 * i + 1;
  }
  if (flag == strmFlag::sameFlag) {
    HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));
  } else if (flag == strmFlag::diffFlag) {
    HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamDefault));
  } else if (flag == strmFlag::diffPrio) {
    int minPriority = 0, maxPriority = 0;
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
    HIP_CHECK(hipStreamCreateWithPriority(&stream1, hipStreamDefault, minPriority));
    HIP_CHECK(hipStreamCreateWithPriority(&stream2, hipStreamDefault, maxPriority));
  } else {
    HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamDefault));
    HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamDefault));
  }
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  bool verifyStreamSync = false;
  SECTION("Verify after hipGraphExecDestroy()") { verifyStreamSync = false; }
  SECTION("Verify after hipStreamSynchronize()") { verifyStreamSync = true; }
  ret = CaptureStreamAndLaunchGraph(A_d, B_d, C_d, A_h.data(), B_h.data(), C_h.data(),
                                    hipStreamCaptureModeGlobal, stream1, stream2, graph,
                                    verifyStreamSync);
  REQUIRE(ret == true);

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

/**
 * Test Description
 * ------------------------
 *    - Start capture of a graph in one thread and end the capture in another
 * thread.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
static void threadCaptureEnd(hipStream_t* streamCapt, hipGraph_t* graph, int* A_d, int* B_d,
                             int* C_d, int* C_h, size_t N) {
  size_t Nbytes = N * sizeof(int);
  HipTest::vectorSUB<<<dim3(blocks), dim3(threadsPerBlock), 0, *streamCapt>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, *streamCapt));
  HIP_CHECK(hipStreamEndCapture(*streamCapt, graph));
}

static void threadCaptureStart(hipStream_t* streamCapt, hipStream_t* streamFork, hipGraph_t* graph,
                               int* A_d, int* B_d, int* C_d, int* A_h, int* B_h, size_t N) {
  size_t Nbytes = N * sizeof(int);
  hipEvent_t e;
  HIP_CHECK(hipEventCreate(&e));
  HIP_CHECK(hipStreamBeginCaptureToGraph(*streamCapt, *graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeRelaxed));
  HIP_CHECK(hipEventRecord(e, *streamCapt));
  HIP_CHECK(hipStreamWaitEvent(*streamFork, e, 0));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, *streamCapt));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, *streamCapt));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, *streamFork));
  HIP_CHECK(hipEventRecord(e, *streamFork));
  HIP_CHECK(hipStreamWaitEvent(*streamCapt, e, 0));
}

TEST_CASE("Unit_hipStreamBeginCaptureToGraph_CapturePartialInThreads") {
  int *A_d, *B_d, *C_d;
  std::vector<int> A_h(N), B_h(N), C_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream1, stream2;
  hipGraph_t graph{nullptr};

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
    B_h[i] = N - i - 1;
  }

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  // Launch threads to capture graph
  std::thread startCaptureThread(threadCaptureStart, &stream1, &stream2, &graph, A_d, B_d, C_d,
                                 A_h.data(), B_h.data(), N);
  startCaptureThread.join();
  std::thread endCaptureThread(threadCaptureEnd, &stream1, &graph, A_d, B_d, C_d, C_h.data(), N);
  endCaptureThread.join();
  // Instantiate and execute the graph
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (A_h[i] - B_h[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

/**
 * Test Description
 * ------------------------
 *    - Capture and execute 2 independent graphs concurrently.
 * ------------------------
 *    - catch\unit\graph\hipStreamBeginCaptureToGraph.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
#ifdef __linux__
// Currently disabled due to defect raised.
static std::atomic<int> retValG(1);
void threadCaptureExec(int* A_d, int* B_d, int* C_d, int* A_h, int* B_h, int* C_h,
                       hipStream_t* stream1, hipStream_t* stream2, hipGraph_t* graph,
                       bool verifyStreamSync) {
  bool ret = false;
  ret = CaptureStreamAndLaunchGraph(A_d, B_d, C_d, A_h, B_h, C_h, hipStreamCaptureModeRelaxed,
                                    *stream1, *stream2, *graph, verifyStreamSync);
  int val = 0;
  if (ret) {
    val = 1;
  }
  retValG.fetch_and(val);
}

TEST_CASE("Unit_hipStreamBeginCaptureToGraph_IndepGraphsThreads") {
  int *A1_d, *B1_d, *C1_d;
  std::vector<int> A1_h(N), B1_h(N), C1_h(N);
  int *A2_d, *B2_d, *C2_d;
  std::vector<int> A2_h(N), B2_h(N), C2_h(N);
  size_t Nbytes = N * sizeof(int);
  hipStream_t stream1, stream2, stream3, stream4;
  hipGraph_t graph1{nullptr}, graph2{nullptr};

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A2_h[i] = A1_h[i] = 2 * i;
    B2_h[i] = B1_h[i] = 2 * i + 1;
  }

  // Create stream and empty graph
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipStreamCreate(&stream4));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  // Allocate device resources
  HIP_CHECK(hipMalloc(&A1_d, Nbytes));
  HIP_CHECK(hipMalloc(&B1_d, Nbytes));
  HIP_CHECK(hipMalloc(&C1_d, Nbytes));
  HIP_CHECK(hipMalloc(&A2_d, Nbytes));
  HIP_CHECK(hipMalloc(&B2_d, Nbytes));
  HIP_CHECK(hipMalloc(&C2_d, Nbytes));

  // Capture an independent graph from stream
  bool verifyStreamSync = false;

  SECTION("Verify after hipGraphExecDestroy()") { verifyStreamSync = false; }

  SECTION("Verify after hipStreamSynchronize()") { verifyStreamSync = true; }

  std::thread thread1(threadCaptureExec, A1_d, B1_d, C1_d, A1_h.data(), B1_h.data(), C1_h.data(),
                      &stream1, &stream2, &graph1, verifyStreamSync);

  std::thread thread2(threadCaptureExec, A2_d, B2_d, C2_d, A2_h.data(), B2_h.data(), C2_h.data(),
                      &stream3, &stream4, &graph2, verifyStreamSync);
  thread1.join();
  thread2.join();

  REQUIRE(retValG.load() == 1);
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream4));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipFree(A1_d));
  HIP_CHECK(hipFree(B1_d));
  HIP_CHECK(hipFree(C1_d));
  HIP_CHECK(hipFree(A2_d));
  HIP_CHECK(hipFree(B2_d));
  HIP_CHECK(hipFree(C2_d));
  fprintf(stderr, "Unit_hipStreamBeginCaptureToGraph_IndepGraphsThreads\n");
}
#endif
