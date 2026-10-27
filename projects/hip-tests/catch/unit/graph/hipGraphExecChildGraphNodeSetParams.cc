/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
  Testcase Scenarios of hipGraphExecChildGraphNodeSetParams API:

  Functional Scenarios:
  1. Create child graph, Instantiate the graph and update the child graph
     using hipGraphExecChildGraphNodeSetParams API
  2. Create child graph with topology, Instantiate the graph
     and update the child graph
     using hipGraphExecChildGraphNodeSetParams API

  Negative Scenarios:
  1. Pass nullptr to child graph
  2. Pass nullptr to graphnode
  3. Pass nullptr to graphExec
  4. Pass uninitialized graph node
  5. Pass orginial graph node instead of child graph node
  6. Change topology of child graph node

  Testcase Scenarios of updating nodes inside a child graph through the parent graphExec:

  Functional Scenarios:
  1. Update an embedded memset node using hipGraphExecMemsetNodeSetParams
  2. Update an embedded memcpy node using hipGraphExecMemcpyNodeSetParams1D
  3. Update an embedded kernel node using hipGraphExecKernelNodeSetParams
  4. Update an embedded memset node using the generic hipGraphExecNodeSetParams

  Negative Scenarios:
  1. Pass the caller's own child graph nodes instead of the embedded ones
 **/

#include <vector>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_Negative) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1, childgraph2;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<hipGraphNode_t> childdependencies;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_A, memcpyH2D_B_child, childGraphNode1;
  HIP_CHECK(hipMemcpy(C_d, C_h, Nbytes, hipMemcpyHostToDevice));

  // Adding MemcpyNode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  // Adding memcpyNode to childgraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, nullptr, 0, B_d, A_d, Nbytes,
                                    hipMemcpyDeviceToDevice));

  // Adding childnode to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph, nullptr, 0, childgraph1));

  // Adding memcpynode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0, B_h, B_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &childGraphNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyD2H_A, 1));

  // Adding memcpynode to new childgraph which is used to update the
  // childgraph node
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B_child, childgraph2, nullptr, 0, B_d, C_d, Nbytes,
                                    hipMemcpyDeviceToDevice));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Pass nullptr to graphExec") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(nullptr, childGraphNode1, childgraph2) ==
            hipErrorInvalidValue);
  }
  SECTION("Pass nullptr to child graph node") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec, nullptr, childgraph2) ==
            hipErrorInvalidValue);
  }
  SECTION("Pass nullptr to child graph") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode1, nullptr) ==
            hipErrorInvalidValue);
  }
  SECTION("Passing parent graph instead of child graph") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode1, graph) != hipSuccess);
  }
  SECTION("Updating the child graph topology") {
    hipGraphNode_t newnode;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&newnode, childgraph2, nullptr, 0, B_d, C_d, Nbytes,
                                      hipMemcpyDeviceToDevice));
    HIP_CHECK(hipGraphAddDependencies(childgraph2, &memcpyH2D_B_child, &newnode, 1));

    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode1, childgraph2) !=
            hipSuccess);
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
   This testcase verifies the following scenario
   Create graph, add child node to graph, Instantiate the graph
   and update the child graph node with a new graph
   using hipGraphExecChildGraphNodeSetParams API
   and execute it
   */
HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1, childgraph2;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<hipGraphNode_t> childdependencies;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_A, memcpyH2D_B_child, childGraphNode1;
  HIP_CHECK(hipMemcpy(C_d, C_h, Nbytes, hipMemcpyHostToDevice));

  // Adding MemcpyNode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  // Adding memcpyNode to childgraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, nullptr, 0, B_d, A_d, Nbytes,
                                    hipMemcpyDeviceToDevice));

  // Adding childnode to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph, nullptr, 0, childgraph1));

  // Adding memcpynode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0, B_h, B_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &childGraphNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyD2H_A, 1));

  // Adding memcpynode to new childgraph which is used to update the
  // childgraph node
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B_child, childgraph2, nullptr, 0, B_d, C_d, Nbytes,
                                    hipMemcpyDeviceToDevice));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Update the childgraph node
  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode1, childgraph2));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  for (unsigned int i = 0; i < N; i++) {
    if (B_h[i] != C_h[i]) {
      WARN("Validation failed " << B_h[i] << "\t" << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
   This testcase verifies the following scenario
   Create graph, Create child graph with a topology and
   add child node to graph, Instantiate the graph
   and update the child graph node with a new graph
   using hipGraphExecChildGraphNodeSetParams API
   and execute it
   */
HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  size_t NElem{N};
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, childgraph1, childgraph2;
  hipGraphExec_t graphExec;
  hipKernelNodeParams kernelNodeParams{};
  hipGraphNode_t kernel_vecAdd;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  std::vector<hipGraphNode_t> childdependencies, childdependencies1;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyH2D_C, childGraphNode1, memcpyD2H_A, memcpyD2D_AB;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));

  // Adding memcpy node to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  // Adding memcpy node to child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D_AB, childgraph1, nullptr, 0, B_d, A_d, Nbytes,
                                    hipMemcpyDeviceToDevice));
  childdependencies.push_back(memcpyD2D_AB);

  // Adding memcpy node to child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, childdependencies.data(),
                                    childdependencies.size(), B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  // Adding memcpy node to child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, childgraph1, childdependencies.data(),
                                    childdependencies.size(), C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));

  childdependencies.clear();
  childdependencies.push_back(memcpyH2D_B);
  childdependencies.push_back(memcpyH2D_C);

  void* kernelArgs2[] = {&B_d, &C_d, &A_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;

  // Adding kernel node to child graph
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, childgraph1, childdependencies.data(),
                                  childdependencies.size(), &kernelNodeParams));

  // Adding child node to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph, nullptr, 0, childgraph1));

  // Adding memcpy node to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0, A_h, A_d, Nbytes,
                                    hipMemcpyDeviceToHost));


  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &childGraphNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyD2H_A, 1));

  // Creating another child graph for updating parameters with the same topology
  // and passing the new child graph to hipGraphExecChildGraphNodeSetParams API
  hipGraphNode_t memcpyD2D_AB1, memcpyH2D_B1, memcpyH2D_C1, kernel_vecAdd1;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D_AB1, childgraph2, nullptr, 0, B_d, A_d, Nbytes,
                                    hipMemcpyDeviceToDevice));
  childdependencies.clear();
  childdependencies.push_back(memcpyD2D_AB1);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B1, childgraph2, childdependencies.data(),
                                    childdependencies.size(), B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C1, childgraph2, childdependencies.data(),
                                    childdependencies.size(), C_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  childdependencies.clear();
  childdependencies.push_back(memcpyH2D_B1);
  childdependencies.push_back(memcpyH2D_C1);

  void* kernelArgs21[] = {&B_d, &C_d, &A_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs21);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd1, childgraph2, childdependencies.data(),
                                  childdependencies.size(), &kernelNodeParams));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode1, childgraph2));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify child graph execution result
  HipTest::checkVectorADD(B_h, B_h, A_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/* Returns the single node of wantType in graph. The child graph below has exactly one
   node of each type, so a second match means the test graph was changed, not the runtime. */
static hipGraphNode_t FindNodeOfType(hipGraph_t graph, hipGraphNodeType wantType) {
  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(numNodes > 0);
  std::vector<hipGraphNode_t> nodes(numNodes);
  HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &numNodes));

  hipGraphNode_t found = nullptr;
  for (size_t i = 0; i < numNodes; ++i) {
    hipGraphNodeType type;
    HIP_CHECK(hipGraphNodeGetType(nodes[i], &type));
    if (type == wantType) {
      REQUIRE(found == nullptr);
      found = nodes[i];
    }
  }
  REQUIRE(found != nullptr);
  return found;
}

static void VerifyResult(const std::vector<int>& result, const std::vector<int>& expected,
                         const char* stage) {
  for (size_t i = 0; i < expected.size(); ++i) {
    if (result[i] != expected[i]) {
      WARN("Validation failed at '" << stage << "' index " << i << ": got " << result[i]
                                    << " expected " << expected[i]);
      REQUIRE(false);
    }
  }
}

/*
   Update individual nodes *inside* a child graph through the parent's executable graph,
   using the node handles of the graph embedded in the parent's child graph node
   (obtained with hipGraphChildGraphNodeGetGraph).

   parent:  memcpyH2D_A (A_h->A_d)   memcpyH2D_S (S_h->S_d)
                           \            /
                          childGraphNode
                                 |
                          memcpyD2H (D_d->result_h)

   child:   memset (B_d=val)        memcpyD2D (A_d->C_d)
                           \            /
                   kernel vectorADD(B_d, C_d, D_d)

   One section per hipGraphExec*NodeSetParams entry point. Catch2 re-runs the setup for
   each section, so every section starts from the same baseline D_d = kMemsetVal0 + A_h.
   Each section checks what the device computed: an update that resolved but rewrote
   nothing would still return hipSuccess.
 */
HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_UpdateEmbeddedNodes) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  constexpr int kMemsetVal0 = 5;
  constexpr int kMemsetVal1 = 11;
  constexpr int kMemsetVal2 = 23;
  size_t NElem{N};
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr}, *D_d{nullptr}, *S_d{nullptr};
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));
  HIP_CHECK(hipMalloc(&S_d, Nbytes));

  std::vector<int> A_h(N), S_h(N), result_h(N), expected(N);
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = static_cast<int>(i) + 1;
    S_h[i] = static_cast<int>(i) + 1000;
  }

  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  // Child graph: two roots joining on the kernel.
  hipGraph_t childGraph;
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  hipMemsetParams memsetParams{};
  memsetParams.dst = B_d;
  memsetParams.elementSize = sizeof(int);
  memsetParams.width = N;
  memsetParams.height = 1;
  memsetParams.pitch = 0;
  memsetParams.value = kMemsetVal0;

  hipGraphNode_t childMemset, childMemcpy, childKernel;
  HIP_CHECK(hipGraphAddMemsetNode(&childMemset, childGraph, nullptr, 0, &memsetParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&childMemcpy, childGraph, nullptr, 0, C_d, A_d, Nbytes,
                                    hipMemcpyDeviceToDevice));

  void* kernelArgs[] = {&B_d, &C_d, &D_d, reinterpret_cast<void*>(&NElem)};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;

  std::vector<hipGraphNode_t> childDeps{childMemset, childMemcpy};
  HIP_CHECK(hipGraphAddKernelNode(&childKernel, childGraph, childDeps.data(), childDeps.size(),
                                  &kernelNodeParams));

  // Parent graph: two roots joining on the child node.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyH2D_A, memcpyH2D_S, childGraphNode, memcpyD2H;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_S, graph, nullptr, 0, S_d, S_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));

  std::vector<hipGraphNode_t> parentDeps{memcpyH2D_A, memcpyH2D_S};
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, parentDeps.data(), parentDeps.size(),
                                      childGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, &childGraphNode, 1, result_h.data(), D_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + A_h[i];
  VerifyResult(result_h, expected, "baseline");

  // The parent holds a clone of the caller's graph, not the caller's graph itself.
  hipGraph_t embeddedGraph = nullptr;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode, &embeddedGraph));
  REQUIRE(embeddedGraph != nullptr);
  REQUIRE(embeddedGraph != childGraph);

  hipGraphNode_t embMemset = FindNodeOfType(embeddedGraph, hipGraphNodeTypeMemset);
  hipGraphNode_t embMemcpy = FindNodeOfType(embeddedGraph, hipGraphNodeTypeMemcpy);
  hipGraphNode_t embKernel = FindNodeOfType(embeddedGraph, hipGraphNodeTypeKernel);

  SECTION("Update embedded memset node with hipGraphExecMemsetNodeSetParams") {
    memsetParams.value = kMemsetVal1;
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(graphExec, embMemset, &memsetParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal1 + A_h[i];
    VerifyResult(result_h, expected, "memset updated");
  }

  SECTION("Update embedded memcpy node with hipGraphExecMemcpyNodeSetParams1D") {
    // Switch the source from A_d to S_d; the kind must stay DeviceToDevice.
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graphExec, embMemcpy, C_d, S_d, Nbytes,
                                                hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + S_h[i];
    VerifyResult(result_h, expected, "memcpy1D updated");
  }

  SECTION("Update embedded kernel node with hipGraphExecKernelNodeSetParams") {
    // Only the arguments change: both inputs become B_d, so D_d = 2 * kMemsetVal0.
    void* kernelArgsUpdated[] = {&B_d, &B_d, &D_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsUpdated);
    HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, embKernel, &kernelNodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + kMemsetVal0;
    VerifyResult(result_h, expected, "kernel updated");
  }

  SECTION("Update embedded memset node with generic hipGraphExecNodeSetParams") {
    hipGraphNodeParams nodeParams{};
    nodeParams.type = hipGraphNodeTypeMemset;
    nodeParams.memset = memsetParams;
    nodeParams.memset.value = kMemsetVal2;
    HIP_CHECK(hipGraphExecNodeSetParams(graphExec, embMemset, &nodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal2 + A_h[i];
    VerifyResult(result_h, expected, "generic node params updated");
  }

  SECTION("Caller's own child graph node handles are rejected") {
    // The child was cloned on add, so these nodes are not part of the parent graph.
    HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(graphExec, childMemset, &memsetParams),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParams1D(graphExec, childMemcpy, C_d, S_d, Nbytes,
                                                      hipMemcpyDeviceToDevice),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, childKernel, &kernelNodeParams),
                    hipErrorInvalidValue);

    // The rejected updates must not have disturbed the graph.
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    VerifyResult(result_h, expected, "after rejected updates");
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(S_d));
}
