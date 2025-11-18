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
 * Functional Test for API - hipGraphNodeSetEnabled
 1) Add MemCpy node to the graph and verify in graphExec it's enabled status
 2) Add MemSet node to the graph and verify in graphExec it's enabled status
 3) Add Kernel node to the graph and verify in graphExec it's enabled status
- Can check other node type, as only above 3 types are mentioned in the doc
 4) Add HostNode node to the graph and verify in graphExec it's enabled status
 5) Add emptyNode node to the graph and verify in graphExec it's enabled status
 6) Add ChildNode node to the graph and verify in graphExec it's enabled status
 7) Add EventWait node to the graph and verify in graphExec it's enabled status
 8) Add EventRecord node to the graph & verify in graphExec it's enabled status
 */

bool static verifyArray(char* A_h, char* C_h, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (C_h[i] != A_h[i]) {
      INFO("Array A and C not matching at " << i);
      return false;
    }
  }
  return true;
}

bool static verifyVectorSquare(int* A_h, int* C_h, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      INFO("VectorSquare A and C not matching at " << i);
      return false;
    }
  }
  return true;
}

static void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < N; i++) {
    A[i] = i;
  }
}

TEST_CASE("Unit_hipGraphNodeSetEnabled_Functional_Basic") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, childGraph;
  hipGraphNode_t memcpy_A, memcpy_B, memsetNode, kNodeAdd;
  hipGraphNode_t hostNode, emptyNode, childGraphNode, eventWait, eventRecord;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};
  unsigned int isEnabled, setEnable;
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

  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Create graphExec with Kernel node and verify its enabled status") {
    setEnable = 0;
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, kNodeAdd, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, kNodeAdd, &isEnabled));
    REQUIRE(setEnable == isEnabled);

    setEnable = 1;
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, kNodeAdd, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, kNodeAdd, &isEnabled));
    REQUIRE(setEnable == isEnabled);
  }
  SECTION("Create graphExec with MemCpy node and verify its enabled status") {
    setEnable = 0;
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memcpy_A, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memcpy_A, &isEnabled));
    REQUIRE(setEnable == isEnabled);

    setEnable = 1;
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memcpy_A, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memcpy_A, &isEnabled));
    REQUIRE(setEnable == isEnabled);
  }
  SECTION("Create graphExec with MemSet node and verify its enabled status") {
    setEnable = 0;
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memsetNode, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memsetNode, &isEnabled));
    REQUIRE(setEnable == isEnabled);

    setEnable = 1;
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memsetNode, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memsetNode, &isEnabled));
    REQUIRE(setEnable == isEnabled);
  }
  setEnable = 1;
  SECTION("Create graphExec with hostNode and verify its enabled status") {
    ret = hipGraphNodeSetEnabled(graphExec, hostNode, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with emptyNode and verify its enabled status") {
    ret = hipGraphNodeSetEnabled(graphExec, emptyNode, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with ChildGraphNode & verify its enabled status") {
    ret = hipGraphNodeSetEnabled(graphExec, childGraphNode, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with EventWait and verify its enabled status") {
    ret = hipGraphNodeSetEnabled(graphExec, eventWait, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec with EventRecord and verify its enabled status") {
    ret = hipGraphNodeSetEnabled(graphExec, eventRecord, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/* Functional Test for API - hipGraphNodeSetEnabled
 9) Add a Kernel node(Vector_Square) with functonallty to the graph and than
    Disable kernel node and verify the result and
 10) Enable Kernel node and verify the result
 11) Make ClonedGraph and disabled the kernel node in ClonedGraph and verify
 12) Enable the Kernel node in ClonedGraph and verify the result */

TEST_CASE("Unit_hipGraphNodeSetEnabled_Functional_KernelNode") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, clonedGraph;
  hipStream_t stream;
  hipGraphNode_t memcpy_A, memcpy_C, kNodeSquare, kNodeSquare_C;
  hipKernelNodeParams kNodeParams{};
  int *A_d, *C_d, *A_h, *C_h;
  hipGraphExec_t graphExec, clonedGraphExec;
  size_t NElem{N};
  unsigned int setEnable = 0, getEnable = 0;

  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeSquare, graph, nullptr, 0, &kNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kNodeSquare, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNodeSquare, &memcpy_C, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  //  Verify the execution result - basic check 1st time
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(true == verifyVectorSquare(A_h, C_h, N));

  SECTION("After disabled kernel node and verify the execution result") {
    setEnable = 0;  // for disabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, kNodeSquare, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, kNodeSquare, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipMemset(C_d, 0, Nbytes));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true != verifyVectorSquare(A_h, C_h, N));
  }
  SECTION("Again enabled kernel node and verify the execution result") {
    setEnable = 1;  // for enabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, kNodeSquare, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, kNodeSquare, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipMemset(C_d, 0, Nbytes));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyVectorSquare(A_h, C_h, N));
  }

  // Code for cloned graph operations
  HIP_CHECK(hipGraphClone(&clonedGraph, graph));
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphNodeFindInClone(&kNodeSquare_C, kNodeSquare, clonedGraph));

  HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(true == verifyVectorSquare(A_h, C_h, N));

  SECTION("After disabled kernel node in ClonedGraph & verify the result") {
    setEnable = 0;  // for disabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(clonedGraphExec, kNodeSquare_C, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(clonedGraphExec, kNodeSquare_C, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipMemset(C_d, 0, Nbytes));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true != verifyVectorSquare(A_h, C_h, N));
  }
  SECTION("Again enabled kernel node in ClonedGraph & verify the result") {
    setEnable = 1;  // for enabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(clonedGraphExec, kNodeSquare_C, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(clonedGraphExec, kNodeSquare_C, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipMemset(C_d, 0, Nbytes));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyVectorSquare(A_h, C_h, N));
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/* Functional Test for API - hipGraphNodeSetEnabled
 13) Add a MemSet node with functonally to the graph and than
    Disable MemSet node and verify the result and
 14) Enable MemSet node and verify the result
 15) Make ClonedGraph and disabled the MemSet node in ClonedGraph and verify
 16) Enable the MemSet node in ClonedGraph and verify the result */

TEST_CASE("Unit_hipGraphNodeSetEnabled_Functional_MemSet") {
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 9;
  hipGraph_t graph, clonedGraph;
  hipGraphExec_t graphExec, clonedGraphExec;
  hipStream_t stream;
  hipGraphNode_t memcpy_A, memcpy_AC, memcpy_C, memsetNode, memsetNode_C;
  unsigned int setEnable = 0, getEnable = 0;
  char *A_d, *C_d, *A_h, *B_h, *C_h;

  HipTest::initArrays<char>(&A_d, nullptr, &C_d, &A_h, &B_h, &C_h, N, false);

  memset(B_h, val, Nbytes);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = val;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_AC, graph, nullptr, 0, C_d, A_d, Nbytes,
                                    hipMemcpyDeviceToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &memsetNode, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetNode, &memcpy_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_AC, &memcpy_C, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  //  Verify the execution result - basic check 1st time
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(true == verifyArray(B_h, C_h, N));

  SECTION("After disabled MemSet node and verify the execution result") {
    setEnable = 0;  // for disabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memsetNode, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memsetNode, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyArray(A_h, C_h, N));
  }
  SECTION("Again enabled MemSet node and verify the execution result") {
    setEnable = 1;  // for enabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memsetNode, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memsetNode, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyArray(B_h, C_h, N));
  }

  // Code for cloned graph operations
  HIP_CHECK(hipGraphClone(&clonedGraph, graph));
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphNodeFindInClone(&memsetNode_C, memsetNode, clonedGraph));

  HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(true == verifyArray(B_h, C_h, N));

  SECTION("After disabled MemSet node and verify the execution result") {
    setEnable = 0;  // for disabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(clonedGraphExec, memsetNode_C, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(clonedGraphExec, memsetNode_C, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyArray(A_h, C_h, N));
  }
  SECTION("Again enabled MemSet node and verify the execution result") {
    setEnable = 1;  // for enabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(clonedGraphExec, memsetNode_C, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(clonedGraphExec, memsetNode_C, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyArray(B_h, C_h, N));
  }

  HipTest::freeArrays<char>(A_d, nullptr, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/* Functional Test for API - hipGraphNodeSetEnabled
 17) Add a MemCpy node with functonally to the graph and than
    Disable MemCpy node and verify the result and
 18) Enable MemCpy node and verify the result
 19) Make ClonedGraph and disabled the MemCpy node in ClonedGraph and verify
 20) Enable the MemCpy node in ClonedGraph and verify the result */

TEST_CASE("Unit_hipGraphNodeSetEnabled_Functional_MemCpy") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, clonedGraph;
  hipStream_t stream;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kNodeSquare, memcpy_B_C;
  hipKernelNodeParams kNodeParams{};
  int *A_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec, clonedGraphExec;
  size_t NElem{N};
  unsigned int setEnable = 0, getEnable = 0;

  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, A_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeSquare, graph, nullptr, 0, &kNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &memcpy_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kNodeSquare, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNodeSquare, &memcpy_C, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  //  Verify the execution result - basic check 1st time
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(true == verifyVectorSquare(B_h, C_h, N));

  SECTION("After disabled MemCpy node and verify the execution result") {
    setEnable = 0;  // for disabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memcpy_B, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memcpy_B, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyVectorSquare(A_h, C_h, N));
  }
  SECTION("Again enabled MemCpy node and verify the execution result") {
    setEnable = 1;  // for enabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(graphExec, memcpy_B, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(graphExec, memcpy_B, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyVectorSquare(B_h, C_h, N));
  }

  // Code for cloned graph operations
  HIP_CHECK(hipGraphClone(&clonedGraph, graph));
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphNodeFindInClone(&memcpy_B_C, memcpy_B, clonedGraph));

  HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(true == verifyVectorSquare(B_h, C_h, N));

  SECTION("After disabled MemSet node and verify the execution result") {
    setEnable = 0;  // for disabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(clonedGraphExec, memcpy_B_C, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(clonedGraphExec, memcpy_B_C, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyVectorSquare(A_h, C_h, N));
  }
  SECTION("Again enabled MemSet node and verify the execution result") {
    setEnable = 1;  // for enabled a node
    HIP_CHECK(hipGraphNodeSetEnabled(clonedGraphExec, memcpy_B_C, setEnable));
    HIP_CHECK(hipGraphNodeGetEnabled(clonedGraphExec, memcpy_B_C, &getEnable));
    REQUIRE(setEnable == getEnable);

    HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(true == verifyVectorSquare(B_h, C_h, N));
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 Negative Test for API - hipGraphNodeSetEnabled
 1) Pass graphExec as nullptr
 2) Pass graphExec as uninitialized object
 3) Pass Node as nullptr
 4) Pass Node as uninitialized object
 5) Pass setEnable as INT_MAX

 Negative Functional Test for API - hipGraphNodeSetEnabled
 6) Pass hNode from different graph and verify
 7) Create graphExec and then add one more new node to the graph verify
 8) Pass hNode as clonedGraph node and exec as graphExec
 9) Pass graphExec as clonedGraphExec and hNode as graph node
 10) Pass hNode a deleted node from same graph where exec was created
 11) Create graphExec and then delete the graph and verify a node
 12) Create graphExec and then delete the graphExec and verify a node
 */

TEST_CASE("Unit_hipGraphNodeSetEnabled_Negative_Functional") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, graph2, clonedGraph;
  hipGraphNode_t memcpy_A, memcpy_A_C, memcpy_B, memcpy_C, memcpy_A2, kNodeAdd;
  hipKernelNodeParams kNodeParams{};
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec, graphExec2, clonedGraphExec;
  size_t NElem{N};
  unsigned int setEnable = 1;
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

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph, nullptr, nullptr, 0));

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
    ret = hipGraphNodeSetEnabled(nullptr, memcpy_B, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hGraphExec as uninitialized object") {
    hipGraphExec_t graphExec_uninit{};
    ret = hipGraphNodeSetEnabled(graphExec_uninit, memcpy_B, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Node as nullptr") {
    ret = hipGraphNodeSetEnabled(graphExec, nullptr, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Node as uninitialized object") {
    hipGraphNode_t node_uninit{};
    ret = hipGraphNodeSetEnabled(graphExec, node_uninit, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass setEnable as INT_MAX") {
    ret = hipGraphNodeSetEnabled(graphExec, memcpy_B, INT_MAX);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass hNode from different graph and verify") {
    ret = hipGraphNodeSetEnabled(graphExec, memcpy_A2, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Create graphExec and add one more new node to the graph & verify") {
    ret = hipGraphNodeSetEnabled(graphExec, memcpy_C, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hNode as clonedGraph node and exec as graphExec") {
    ret = hipGraphNodeSetEnabled(graphExec, memcpy_A_C, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass graphExec as clonedGraphExec and hNode as graph node") {
    ret = hipGraphNodeSetEnabled(clonedGraphExec, memcpy_A, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hNode a deleted node from same graph where exec was created") {
    HIP_CHECK(hipGraphDestroyNode(memcpy_A));
    ret = hipGraphNodeSetEnabled(graphExec, memcpy_A, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  SECTION("Create graphExec and then delete the graphExec and verify a node") {
    ret = hipGraphNodeSetEnabled(graphExec2, memcpy_B, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipGraphDestroy(graph));
  SECTION("Create graphExec and then delete the graph and verify a node") {
    ret = hipGraphNodeSetEnabled(graphExec, memcpy_B, setEnable);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(graph2));
}
