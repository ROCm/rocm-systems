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

#define N 1024

/**
 * Functional Test for API - hipGraphNodeGetEnabled
 1) Add MemCpy node to the graph and verify in graphExec it enabled status
 2) Add MemSet node to the graph and verify in graphExec it enabled status
 3) Add Kernel node to the graph and verify in graphExec it enabled status
- Can check other node type, as only above 3 types are mentioned in the doc
 4) Add HostNode node to the graph and verify in graphExec it enabled status
 5) Add emptyNode node to the graph and verify in graphExec it enabled status
 6) Add ChildNode node to the graph and verify in graphExec it enabled status
 7) Add EventWait node to the graph and verify in graphExec it enabled status
 8) Add EventRecord node to the graph and verify in graphExec it enabled status
 */

static void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < N; i++) {
    A[i] = i;
  }
}

TEST_CASE("Unit_hipGraphNodeGetEnabled_Functional_Basic") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, childGraph;
  hipGraphNode_t memcpy_A, memcpy_B, memsetNode, kNodeAdd;
  hipGraphNode_t hostNode, emptyNode, childGraphNode, eventWait, eventRecord;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};
  unsigned int isEnabled = 0;
  hipError_t ret;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  hipKernelNodeParams kNodeParams{};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeAdd, graph, nullptr, 0, &kNodeParams));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 7;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));

  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipGraphAddEventRecordNode(&eventRecord, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphAddEventWaitNode(&eventWait, graph, nullptr, 0, event));

  HIP_CHECK(hipGraphCreate(&childGraph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, childGraph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  // Adding child node to clonedGraph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  SECTION("Create graphExec with Kernel node and verify its enabled status") {
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, kNodeAdd, &isEnabled));
    REQUIRE(1 == isEnabled);
  }
  SECTION("Create graphExec with MemCpy node and verify its enabled status") {
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memcpy_A, &isEnabled));
    REQUIRE(1 == isEnabled);
  }
  SECTION("Create graphExec with MemSet node and verify its enabled status") {
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memsetNode, &isEnabled));
    REQUIRE(1 == isEnabled);
  }
  SECTION("Create graphExec with hostNode and verify its enabled status") {
    ret = hipGraphNodeGetEnabled(graphExec, hostNode, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with emptyNode and verify its enabled status") {
    ret = hipGraphNodeGetEnabled(graphExec, emptyNode, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with ChildGraphNode & verify its enabled status") {
    ret = hipGraphNodeGetEnabled(graphExec, childGraphNode, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with EventWait and verify its enabled status") {
    ret = hipGraphNodeGetEnabled(graphExec, eventWait, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with EventRecord and verify its enabled status") {
    ret = hipGraphNodeGetEnabled(graphExec, eventRecord, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 Negative Test for API - hipGraphNodeGetEnabled
 1) Pass graphExec as nullptr
 2) Pass graphExec as uninitialized object
 3) Pass Node as nullptr
 4) Pass Node as uninitialized object
 5) Pass isEnabled as nullptr

 Negative Functional Test for API - hipGraphNodeGetEnabled
 6) Pass hNode from different graph and verify
 7) Create graphExec and then add one more new node to the graph verify
 8) Pass hNode as clonedGraph node and exec as graphExec
 9) Pass graphExec as clonedGraphExec and hNode as graph node
 10) Pass hNode a deleted node from same graph where exec was created
 11) Create graphExec and then delete the graph and verify a node
 12) Create graphExec and then delete the graphExec and verify a node
 */

TEST_CASE("Unit_hipGraphNodeGetEnabled_Negative_Functional") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, graph2, clonedGraph;
  hipGraphNode_t memcpy_A, memcpy_A_C, memcpy_B, memcpy_A2, memcpy_C, kNodeAdd;
  hipKernelNodeParams kNodeParams{};
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec, graphExec2, clonedGraphExec;
  size_t NElem{N};
  unsigned int isEnabled;
  hipError_t ret;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeAdd, graph, nullptr, 0, &kNodeParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kNodeAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kNodeAdd, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph, NULL, NULL, 0));

  HIP_CHECK(hipGraphClone(&clonedGraph, graph));
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphNodeFindInClone(&memcpy_A_C, memcpy_A, clonedGraph));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNodeAdd, &memcpy_C, 1));

  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A2, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  SECTION("Pass hGraphExec as nullptr") {
    ret = hipGraphNodeGetEnabled(nullptr, memcpy_B, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hGraphExec as uninitialized object") {
    hipGraphExec_t graphExec_uninit{};
    ret = hipGraphNodeGetEnabled(graphExec_uninit, memcpy_B, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Node as nullptr") {
    ret = hipGraphNodeGetEnabled(graphExec, nullptr, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Node as uninitialized object") {
    hipGraphNode_t node_uninit{};
    ret = hipGraphNodeGetEnabled(graphExec, node_uninit, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass isEnabled as nullptr") {
    ret = hipGraphNodeGetEnabled(graphExec, memcpy_B, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hNode from different graph and verify") {
    ret = hipGraphNodeGetEnabled(graphExec, memcpy_A2, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec and add one more new node to the graph & verify") {
    ret = hipGraphNodeGetEnabled(graphExec, memcpy_C, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hNode as clonedGraph node and exec as graphExec") {
    ret = hipGraphNodeGetEnabled(graphExec, memcpy_A_C, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass graphExec as clonedGraphExec and hNode as graph node") {
    ret = hipGraphNodeGetEnabled(clonedGraphExec, memcpy_A, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hNode a deleted node from same graph where exec was created") {
    HIP_CHECK(hipGraphDestroyNode(memcpy_A));
    ret = hipGraphNodeGetEnabled(graphExec, memcpy_A, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  SECTION("Create graphExec and then delete the graphExec and verify a node") {
    ret = hipGraphNodeGetEnabled(graphExec2, memcpy_B, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipGraphDestroy(graph));
  SECTION("Create graphExec and then delete the graph and verify a node") {
    ret = hipGraphNodeGetEnabled(graphExec, memcpy_B, &isEnabled);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(graph2));
}
