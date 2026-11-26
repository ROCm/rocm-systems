/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <functional>
#include <vector>

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

#define SIZE 1024
static char memSetVal = 'a';

/**
 * @addtogroup hipGraphAddMemsetNode hipGraphAddMemsetNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddMemsetNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t
 * *pDependencies, size_t numDependencies, const hipMemsetParams *pMemsetParams)` -
 * Creates a memset node and adds it to a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that all elements of destination memory are set to the correct value.
 * The test is repeated for all valid element sizes(1, 2, 4), and several allocations of different
 * height and width, both on host and device.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipGraphAddMemsetNode_Positive_Basic, uint8_t, uint16_t, uint32_t) {
  const auto f = [](hipMemsetParams* params) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, params));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  GraphMemsetNodeCommonPositive<TestType, hipMemsetParams>(f);
}

/**
 * Test Description
 * ------------------------
 *    - Fill rows of a few to a hundred elements of each valid element size (1, 2, 4) with a 2D
 * memset node, in an allocation with a pitch larger than the row, and verify every byte: the rows
 * hold the value and the padding between them is untouched.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipGraphAddMemsetNode_Positive_2DPadding, uint8_t, uint16_t,
                       uint32_t) {
  constexpr uint32_t memsetVal = 0xD1E2F304u;
  constexpr uint8_t paddingVal = 0x3C;
  constexpr size_t numRows = 4;
  const size_t numElements = GENERATE(1, 3, 5, 16, 20, 100);
  const size_t width = numElements * sizeof(TestType);
  const size_t pitch = width + 32;
  const size_t allocSize = pitch * numRows + 64;

  LinearAllocGuard<uint8_t> dst_d(LinearAllocs::hipMalloc, allocSize);
  HIP_CHECK(hipMemset(dst_d.ptr(), paddingVal, allocSize));

  hipMemsetParams params{};
  params.dst = dst_d.ptr();
  params.elementSize = sizeof(TestType);
  params.width = numElements;
  params.height = numRows;
  params.pitch = pitch;
  params.value = memsetVal;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  std::vector<uint8_t> dst_h(allocSize);
  HIP_CHECK(hipMemcpy(dst_h.data(), dst_d.ptr(), allocSize, hipMemcpyDeviceToHost));
  std::vector<uint8_t> expected(allocSize, paddingVal);
  for (size_t row = 0; row < numRows; row++) {
    for (size_t column = 0; column < width; column++) {
      expected[row * pitch + column] =
          static_cast<uint8_t>(memsetVal >> (8 * (column % sizeof(TestType))));
    }
  }
  const auto mismatch = std::mismatch(dst_h.begin(), dst_h.end(), expected.begin());
  INFO("element size " << sizeof(TestType) << " width " << width << ": first mismatch at byte "
                       << (mismatch.first - dst_h.begin()));
  REQUIRE(mismatch.first == dst_h.end());
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *        -# pGraphNode is nullptr
 *        -# graph is nullptr
 *        -# pDependencies is nullptr when numDependencies is not zero
 *        -# A node in pDependencies originates from a different graph
 *        -# numDependencies is invalid
 *        -# A node is duplicated in pDependencies
 *        -# pMemsetParams is nullptr
 *        -# pMemsetParams::dst is nullptr
 *        -# pMemsetParams::elementSize is different from 1, 2, and 4
 *        -# pMemsetParams::width is zero
 *        -# pMemsetParams::width is larger than the allocated memory region
 *        -# pMemsetParams::height is zero
 *        -# pMemsetParams::pitch is less than width when height is more than 1
 *        -# pMemsetParams::pitch * pMemsetParams::height is larger than the allocated memory region
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_Negative_Parameters) {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, 4 * sizeof(int));
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(*alloc.ptr());
  params.width = 1;
  params.height = 1;
  params.value = 42;

  GraphAddNodeCommonNegativeTests(std::bind(hipGraphAddMemsetNode, _1, _2, _3, _4, &params), graph);

  hipGraphNode_t node = nullptr;
  MemsetCommonNegative(std::bind(hipGraphAddMemsetNode, &node, graph, nullptr, 0, _1), params);

  HIP_CHECK(hipGraphDestroy(graph));
}
/*
 * Allocate a 2D array using hipMallocPitch. Initialize the allocated memory
 * using hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
 */
HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_hipMallocPitch_2D) {
  size_t width = SIZE * sizeof(char), numW{SIZE}, numH{SIZE}, pitch_A;
  char* A_d;

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  // Host memory.
  char* A_h = new char[numW * numH];
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      *(A_h + i * numH + j) = ' ';
    }
  }
  // 2D Memory allocation hipMallocPitch
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, numH));
  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;
  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = pitch_A;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = numH;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);
  // Add MemCpy Node
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(A_d, pitch_A, numW, numH);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = make_hipExtent(width, numH, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &myparms));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      REQUIRE(*(A_h + i * numH + j) == memSetVal);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  delete[] A_h;
  HIP_CHECK(hipFree(A_d));
}
/*
 * Allocate a 1D array using hipMallocPitch. Initialize the allocated memory using
 * hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
 */
HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_hipMallocPitch_1D) {
  size_t width = SIZE * sizeof(char), numW{SIZE}, pitch_A;
  char* A_d;

  // Initialize the host memory
  std::vector<char> A_h(numW, ' ');

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  // 1D Memory allocation hipMallocPitch
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, 1));
  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;
  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = pitch_A;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);
  // Add MemCpy Node
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(A_d, pitch_A, numW, 1);
  myparms.dstPtr = make_hipPitchedPtr(A_h.data(), width, numW, 1);
  myparms.extent = make_hipExtent(width, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
}
/*
 * Allocate a 2D array using hipMalloc3D. Initialize the allocated memory using
 * hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
 */
HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_hipMalloc3D_2D) {
  size_t width = SIZE * sizeof(char);
  size_t numW = SIZE, numH = SIZE;

  // Host Memory
  char* A_h = new char[numW * numH];
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      *(A_h + i * numH + j) = ' ';
    }
  }
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  hipPitchedPtr A_d;
  hipExtent extent3D = make_hipExtent(width, numH, 1);

  // Allocate 3D memory.
  HIPCHECK(hipMalloc3D(&A_d, extent3D));

  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;

  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = A_d.ptr;
  memsetParams.value = memSetVal;
  memsetParams.pitch = A_d.pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = numH;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);

  // MemCpy params
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = A_d;
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = make_hipExtent(width, numH, 1);
  myparms.kind = hipMemcpyDeviceToHost;

  // Add MemCpy Node
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      REQUIRE(*(A_h + i * numH + j) == memSetVal);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  delete[] A_h;
  HIP_CHECK(hipFree(A_d.ptr));
}
/*
 * Allocate a 1D array using hipMalloc3D. Initialize the allocated
 * memory using hipGraphAddMemsetNode. Copy the values in device
 * memory to host using hipGraphAddMemcpyNode. Verify the results.
 */
HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_hipMalloc3D_1D) {
  size_t width = SIZE * sizeof(char);
  size_t numW = SIZE;

  // Initialize the host memory
  std::vector<char> A_h(numW, ' ');

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  hipPitchedPtr A_d;
  hipExtent extent1D = make_hipExtent(width, 1, 1);

  // Allocate 3D memory.
  HIPCHECK(hipMalloc3D(&A_d, extent1D));

  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;

  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = A_d.ptr;
  memsetParams.value = memSetVal;
  memsetParams.pitch = A_d.pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);

  // MemCpy params
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = A_d;
  myparms.dstPtr = make_hipPitchedPtr(A_h.data(), width, numW, 1);
  myparms.extent = make_hipExtent(width, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;

  // Add MemCpy Node
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph))
  HIP_CHECK(hipFree(A_d.ptr));
}
/*
 * Allocate a 1D array using hipMalloc. Initialize the allocated memory using
 * hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
 */
HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_hipMalloc_1D) {
  char* A_d;
  size_t NumW = SIZE;
  size_t Nbytes1D = SIZE * sizeof(char);

  // Initialize the host memory
  std::vector<char> A_h(NumW, ' ');

  // Allocate memory to Device pointer
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes1D));

  // Create the graph
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memsetNode, memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Add Memset node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = Nbytes1D;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = NumW;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);
  // Add MemCpy Node
  hipPitchedPtr devPitchedPtr{A_d, Nbytes1D, NumW, 0};
  hipPitchedPtr hostPitchedPtr{A_h.data(), Nbytes1D, NumW, 0};
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = devPitchedPtr;
  myparms.dstPtr = hostPitchedPtr;
  myparms.extent = make_hipExtent(Nbytes1D, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &myparms));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < NumW; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
}

HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_hipMallocManaged) {
  if (!HipTest::isManagedMemorySupportedOnDevice(0)) {
    WARN(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory attribute"
        "so defaulting to system memory.");
  }
  size_t Nbytes1D = SIZE * sizeof(char);
  char* A_d;
  // Initialize the host memory
  std::vector<char> A_h(SIZE, ' ');
  // Device Memory
  HIP_CHECK(hipMallocManaged(&A_d, SIZE * sizeof(char)));
  // Create the graph
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memsetNode, memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Add Memset node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = Nbytes1D;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = SIZE;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  hipPitchedPtr devPitchedPtr{A_d, Nbytes1D, SIZE, 1};
  hipPitchedPtr hostPitchedPtr{A_h.data(), Nbytes1D, SIZE, 1};

  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = devPitchedPtr;
  myparms.dstPtr = hostPitchedPtr;
  myparms.extent = make_hipExtent(Nbytes1D, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < SIZE; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
}

/*
 * A graph memset node targeting another device's hipMalloc buffer works without
 * hipDeviceEnablePeerAccess: the runtime grants peer access implicitly, as it
 * does for copy operations.  Three shapes are tested:
 *   (a) height == 1 (fully linear)
 *   (b) height > 1 with pitch == width * elementSize (no row padding)
 *   (c) a captured hipMemsetAsync that is instantiated and launched as a graph
 */
HIP_TEST_CASE(Unit_hipGraphAddMemsetNode_PeerBufferFromOtherDevice) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
  if (!can_access_peer) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  // Ensure peer access 0->1 is off so a prior test cannot make this one pass.
  HIP_CHECK(hipSetDevice(0));
  { hipError_t e = hipDeviceDisablePeerAccess(1);
    if (e != hipSuccess && e != hipErrorPeerAccessNotEnabled) HIP_CHECK(e);
    static_cast<void>(hipGetLastError()); }
  // Restore device 0 on scope exit (also fires if a REQUIRE inside a SECTION throws).
  struct DevGuard { ~DevGuard() { static_cast<void>(hipSetDevice(0)); } } dg;

  constexpr size_t kN = 1024;  // bytes

  // Allocate the target buffer on device 1 without enabling peer access.
  HIP_CHECK(hipSetDevice(1));
  LinearAllocGuard<uint8_t> dev1_alloc(LinearAllocs::hipMalloc, kN);
  uint8_t* dev1_buf = dev1_alloc.ptr();

  // Stream and graph work are issued from device 0.
  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream0_guard(Streams::created);
  hipStream_t stream0 = stream0_guard.stream();

  SECTION("height == 1 (linear)") {
    // hipGraphAddMemsetNode with height=1 targets the flat fill path that grants
    // peer access to the device-1 allocation.
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;
    hipMemsetParams params{};
    params.dst = reinterpret_cast<void*>(dev1_buf);
    params.value = 0x5A;
    params.elementSize = sizeof(uint8_t);
    params.width = kN;
    params.height = 1;
    params.pitch = kN;
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, stream0));
    HIP_CHECK(hipStreamSynchronize(stream0));

    std::vector<uint8_t> host(kN, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_buf, kN, hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));
    for (size_t i = 0; i < kN; ++i) {
      INFO("byte " << i);
      REQUIRE(host[i] == 0x5Au);
    }
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("height > 1, pitch == width * elementSize (unpitched 2D)") {
    // A 2D shape with pitch == row bytes (no padding) is equivalent to a linear
    // fill and must also grant peer access implicitly.
    constexpr size_t kWidth = 64;         // elements
    constexpr size_t kHeight = kN / kWidth;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;
    hipMemsetParams params{};
    params.dst = reinterpret_cast<void*>(dev1_buf);
    params.value = 0xAB;
    params.elementSize = sizeof(uint8_t);
    params.width = kWidth;
    params.height = kHeight;
    params.pitch = kWidth;  // pitch == width * elementSize (1 byte each)
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, stream0));
    HIP_CHECK(hipStreamSynchronize(stream0));

    std::vector<uint8_t> host(kN, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_buf, kN, hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));
    for (size_t i = 0; i < kN; ++i) {
      INFO("byte " << i);
      REQUIRE(host[i] == 0xABu);
    }
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("captured hipMemsetAsync into peer buffer") {
    // Stream-capturing hipMemsetAsync targeting a peer allocation and launching
    // the graph must succeed and produce the correct fill value.
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream0, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemsetAsync(dev1_buf, 0xCD, kN, stream0));
    HIP_CHECK(hipStreamEndCapture(stream0, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, stream0));
    HIP_CHECK(hipStreamSynchronize(stream0));

    std::vector<uint8_t> host(kN, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_buf, kN, hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));
    for (size_t i = 0; i < kN; ++i) {
      INFO("byte " << i);
      REQUIRE(host[i] == 0xCDu);
    }
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

}

/**
 * End doxygen group GraphTest.
 * @}
 */
