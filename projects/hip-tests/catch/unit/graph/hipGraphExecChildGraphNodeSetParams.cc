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
  5. Update a node nested two child graphs deep, and both nesting levels together

  Negative Scenarios:
  1. Pass the caller's own child graph nodes instead of the embedded ones
  2. Pass an embedded child graph node belonging to an unrelated parent graph

  Testcase Scenarios of child graph node ownership MOVE (AMD only):

  Functional Scenarios:
  1. Move a child graph into the parent and confirm the parent holds the caller's graph
  2. Update the moved child's memset, memcpy and kernel nodes through the parent's
     graphExec using the caller's own node handles, individually and together
  3. Query the moved graph and still see its nodes

  Negative Scenarios:
  1. A moved graph cannot be destroyed, instantiated, re-parented or passed to
     hipGraphExecUpdate independently
 **/

#include <vector>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include "graph_tests_common.hh"


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
  HipTest::checkTest(expected.data(), result_h.data(), N);

  // The parent holds a clone of the caller's graph, not the caller's graph itself.
  hipGraph_t embeddedGraph = nullptr;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode, &embeddedGraph));
  REQUIRE(embeddedGraph != nullptr);
  REQUIRE(embeddedGraph != childGraph);

  hipGraphNode_t embMemset = GraphGetUniqueNodeOfType(embeddedGraph, hipGraphNodeTypeMemset);
  hipGraphNode_t embMemcpy = GraphGetUniqueNodeOfType(embeddedGraph, hipGraphNodeTypeMemcpy);
  hipGraphNode_t embKernel = GraphGetUniqueNodeOfType(embeddedGraph, hipGraphNodeTypeKernel);

  SECTION("Update embedded memset node with hipGraphExecMemsetNodeSetParams") {
    memsetParams.value = kMemsetVal1;
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(graphExec, embMemset, &memsetParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal1 + A_h[i];
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update embedded memcpy node with hipGraphExecMemcpyNodeSetParams1D") {
    // Switch the source from A_d to S_d; the kind must stay DeviceToDevice.
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graphExec, embMemcpy, C_d, S_d, Nbytes,
                                                hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + S_h[i];
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update embedded kernel node with hipGraphExecKernelNodeSetParams") {
    // Only the arguments change: both inputs become B_d, so D_d = 2 * kMemsetVal0.
    void* kernelArgsUpdated[] = {&B_d, &B_d, &D_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsUpdated);
    HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, embKernel, &kernelNodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + kMemsetVal0;
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update embedded memset node with generic hipGraphExecNodeSetParams") {
    hipGraphNodeParams nodeParams{};
    nodeParams.type = hipGraphNodeTypeMemset;
    nodeParams.memset.dst = B_d;
    nodeParams.memset.elementSize = sizeof(int);
    nodeParams.memset.width = N;
    nodeParams.memset.height = 1;
    nodeParams.memset.pitch = 0;
    nodeParams.memset.value = kMemsetVal2;
    HIP_CHECK(hipGraphExecNodeSetParams(graphExec, embMemset, &nodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal2 + A_h[i];
    HipTest::checkTest(expected.data(), result_h.data(), N);
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
    HipTest::checkTest(expected.data(), result_h.data(), N);
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

/*
   Update a node two levels deep through the parent's executable graph.

   parent:      memcpyH2D_A (A_h->A_d) -> outerChildNode -> memcpyD2H (C_d->result_h)
   outerChild:  memset (B_d=val) -> innerChildNode
   innerChild:  kernel vectorADD(A_d, B_d, C_d)

   A one-level child only recurses into ResolveClonedNode once before hitting the
   graphExec's own node map, so it does not cover the multi-level walk or the chained
   per-child lookups. Here the innermost kernel is reached by calling
   hipGraphChildGraphNodeGetGraph twice, and the packet rewrite has to land on the
   innermost child graph node rather than the outer one.
 */
HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_UpdateNestedChildGraphNodes) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  constexpr int kMemsetVal0 = 7;
  constexpr int kMemsetVal1 = 19;
  size_t NElem{N};
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));

  std::vector<int> A_h(N), result_h(N), expected(N);
  for (size_t i = 0; i < N; ++i) A_h[i] = static_cast<int>(i) + 1;

  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  // Innermost graph: one kernel, C_d = A_d + B_d.
  hipGraph_t innerGraph;
  HIP_CHECK(hipGraphCreate(&innerGraph, 0));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;

  hipGraphNode_t innerKernel;
  HIP_CHECK(hipGraphAddKernelNode(&innerKernel, innerGraph, nullptr, 0, &kernelNodeParams));

  // Outer child graph: memset feeding the nested child graph node.
  hipGraph_t outerGraph;
  HIP_CHECK(hipGraphCreate(&outerGraph, 0));

  hipMemsetParams memsetParams{};
  memsetParams.dst = B_d;
  memsetParams.elementSize = sizeof(int);
  memsetParams.width = N;
  memsetParams.height = 1;
  memsetParams.pitch = 0;
  memsetParams.value = kMemsetVal0;

  hipGraphNode_t outerMemset, innerChildNode;
  HIP_CHECK(hipGraphAddMemsetNode(&outerMemset, outerGraph, nullptr, 0, &memsetParams));
  HIP_CHECK(hipGraphAddChildGraphNode(&innerChildNode, outerGraph, &outerMemset, 1, innerGraph));

  // Parent graph.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyH2D_A, outerChildNode, memcpyD2H;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddChildGraphNode(&outerChildNode, graph, &memcpyH2D_A, 1, outerGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, &outerChildNode, 1, result_h.data(), C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  for (size_t i = 0; i < N; ++i) expected[i] = A_h[i] + kMemsetVal0;
  HipTest::checkTest(expected.data(), result_h.data(), N);

  // Walk down two levels of embedded graphs.
  hipGraph_t embeddedOuter = nullptr;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(outerChildNode, &embeddedOuter));
  REQUIRE(embeddedOuter != outerGraph);

  hipGraphNode_t embeddedInnerChildNode =
      GraphGetUniqueNodeOfType(embeddedOuter, hipGraphNodeTypeGraph);
  hipGraph_t embeddedInner = nullptr;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(embeddedInnerChildNode, &embeddedInner));
  REQUIRE(embeddedInner != innerGraph);

  SECTION("Update kernel node at nesting depth two") {
    hipGraphNode_t embeddedKernel = GraphGetUniqueNodeOfType(embeddedInner, hipGraphNodeTypeKernel);
    // Both inputs become B_d, so C_d = 2 * kMemsetVal0 and no longer depends on A_d.
    void* kernelArgsUpdated[] = {&B_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsUpdated);
    HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, embeddedKernel, &kernelNodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + kMemsetVal0;
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update memset node at nesting depth one of the same graph") {
    hipGraphNode_t embeddedMemset = GraphGetUniqueNodeOfType(embeddedOuter, hipGraphNodeTypeMemset);
    memsetParams.value = kMemsetVal1;
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(graphExec, embeddedMemset, &memsetParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = A_h[i] + kMemsetVal1;
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update both nesting levels") {
    hipGraphNode_t embeddedKernel = GraphGetUniqueNodeOfType(embeddedInner, hipGraphNodeTypeKernel);
    hipGraphNode_t embeddedMemset = GraphGetUniqueNodeOfType(embeddedOuter, hipGraphNodeTypeMemset);

    memsetParams.value = kMemsetVal1;
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(graphExec, embeddedMemset, &memsetParams));
    void* kernelArgsUpdated[] = {&B_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsUpdated);
    HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, embeddedKernel, &kernelNodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal1 + kMemsetVal1;
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(outerGraph));
  HIP_CHECK(hipGraphDestroy(innerGraph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

/*
   An embedded child graph node handle belonging to a different parent graph must not
   resolve against this graphExec.

   This is a different rejection path from passing the caller's own child graph nodes.
   Those belong to a plain graph, so the lookup fails immediately when it cannot treat
   their parent as a child graph node. A foreign embedded node does have a child graph
   node as its parent, so the lookup recurses, misses, and has to terminate cleanly
   instead of resolving to the wrong node.
 */
HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_NodeFromUnrelatedGraphRejected) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  constexpr int kMemsetVal = 3;
  size_t NElem{N};
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));

  std::vector<int> A_h(N), result_h(N), expected(N);
  for (size_t i = 0; i < N; ++i) A_h[i] = static_cast<int>(i) + 1;

  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipMemsetParams memsetParams{};
  memsetParams.dst = B_d;
  memsetParams.elementSize = sizeof(int);
  memsetParams.width = N;
  memsetParams.height = 1;
  memsetParams.pitch = 0;
  memsetParams.value = kMemsetVal;

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;

  // Builds parent -> child(memset -> kernel) -> D2H and instantiates it.
  auto buildPipeline = [&](hipGraph_t* childGraph, hipGraph_t* parentGraph,
                           hipGraphNode_t* childNode, hipGraphExec_t* exec, int* resultHost) {
    HIP_CHECK(hipGraphCreate(childGraph, 0));
    hipGraphNode_t cMemset, cKernel;
    HIP_CHECK(hipGraphAddMemsetNode(&cMemset, *childGraph, nullptr, 0, &memsetParams));
    HIP_CHECK(hipGraphAddKernelNode(&cKernel, *childGraph, &cMemset, 1, &kernelNodeParams));

    HIP_CHECK(hipGraphCreate(parentGraph, 0));
    hipGraphNode_t h2d, d2h;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d, *parentGraph, nullptr, 0, A_d, A_h.data(), Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddChildGraphNode(childNode, *parentGraph, &h2d, 1, *childGraph));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h, *parentGraph, childNode, 1, resultHost, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphInstantiate(exec, *parentGraph, nullptr, nullptr, 0));
  };

  hipGraph_t childGraphA, graphA, childGraphB, graphB;
  hipGraphNode_t childNodeA, childNodeB;
  hipGraphExec_t graphExecA, graphExecB;
  std::vector<int> resultB(N);

  buildPipeline(&childGraphA, &graphA, &childNodeA, &graphExecA, result_h.data());
  buildPipeline(&childGraphB, &graphB, &childNodeB, &graphExecB, resultB.data());

  HIP_CHECK(hipGraphLaunch(graphExecA, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  for (size_t i = 0; i < N; ++i) expected[i] = A_h[i] + kMemsetVal;
  HipTest::checkTest(expected.data(), result_h.data(), N);

  // Nodes of graphB's embedded child graph: valid nodes, wrong graphExec.
  hipGraph_t embeddedB = nullptr;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childNodeB, &embeddedB));
  hipGraphNode_t foreignMemset = GraphGetUniqueNodeOfType(embeddedB, hipGraphNodeTypeMemset);
  hipGraphNode_t foreignKernel = GraphGetUniqueNodeOfType(embeddedB, hipGraphNodeTypeKernel);

  HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(graphExecA, foreignMemset, &memsetParams),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExecA, foreignKernel, &kernelNodeParams),
                  hipErrorInvalidValue);

  // graphB's own top level child graph node is equally foreign to graphExecA.
  HIP_CHECK_ERROR(hipGraphExecChildGraphNodeSetParams(graphExecA, childNodeB, childGraphA),
                  hipErrorInvalidValue);

  // The rejected updates must not have disturbed either graph.
  HIP_CHECK(hipGraphLaunch(graphExecA, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  HipTest::checkTest(expected.data(), result_h.data(), N);

  HIP_CHECK(hipGraphLaunch(graphExecB, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  HipTest::checkTest(expected.data(), resultB.data(), N);

  HIP_CHECK(hipGraphExecDestroy(graphExecA));
  HIP_CHECK(hipGraphExecDestroy(graphExecB));
  HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipGraphDestroy(graphB));
  HIP_CHECK(hipGraphDestroy(childGraphA));
  HIP_CHECK(hipGraphDestroy(childGraphB));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

/*
   Child graph node ownership MOVE: the parent takes the caller's graph itself rather than
   cloning it, so the caller's own node handles become nodes of the parent's tree and stay
   usable for per-node updates against the parent's executable graph.

   parent:  memcpyH2D_A (A_h->A_d)   memcpyH2D_S (S_h->S_d)
                           \            /
                       movedChildNode (ownership = Move)
                                 |
                          memcpyD2H (D_d->result_h)

   child:   memset (B_d=val)        memcpyD2D (A_d->C_d)
                           \            /
                   kernel vectorADD(B_d, C_d, D_d)

   The moved graph is owned by the parent and must not be destroyed by the test: it is
   released when the parent graph is destroyed.
 */
// Child graph node ownership was introduced in CUDA 12.9. Older CUDA has neither the
// ownership enum nor the 'ownership' member of cudaChildGraphNodeParams, so this test
// cannot be compiled there at all.
#if HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12090)
HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_MovedChildGraphOwnership) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  constexpr int kMemsetVal0 = 5;
  constexpr int kMemsetVal1 = 11;
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

  // The child graph whose ownership is handed to the parent. Its node handles are kept
  // and used directly for the exec updates below.
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

  // Parent graph.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyH2D_A, memcpyH2D_S, movedChildNode, memcpyD2H;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_S, graph, nullptr, 0, S_d, S_h.data(), Nbytes,
                                    hipMemcpyHostToDevice));

  // Move the child graph into the parent.
  hipGraphNodeParams childNodeParams{};
  childNodeParams.type = hipGraphNodeTypeGraph;
  childNodeParams.graph.graph = childGraph;
  childNodeParams.graph.ownership = hipGraphChildGraphOwnershipMove;
  std::vector<hipGraphNode_t> parentDeps{memcpyH2D_A, memcpyH2D_S};
  HIP_CHECK(hipGraphAddNode(&movedChildNode, graph, parentDeps.data(), parentDeps.size(),
                            &childNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, &movedChildNode, 1, result_h.data(), D_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  // Under MOVE the parent holds the caller's graph itself, not a copy of it.
  hipGraph_t embeddedGraph = nullptr;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(movedChildNode, &embeddedGraph));
  REQUIRE(embeddedGraph == childGraph);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + A_h[i];
  HipTest::checkTest(expected.data(), result_h.data(), N);

  SECTION("Update moved child memset node with the caller's own handle") {
    memsetParams.value = kMemsetVal1;
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(graphExec, childMemset, &memsetParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal1 + A_h[i];
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update moved child memcpy node with the caller's own handle") {
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graphExec, childMemcpy, C_d, S_d, Nbytes,
                                                hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + S_h[i];
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update moved child kernel node with the caller's own handle") {
    void* kernelArgsUpdated[] = {&B_d, &B_d, &D_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsUpdated);
    HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, childKernel, &kernelNodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal0 + kMemsetVal0;
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Update every moved child node in one pass") {
    memsetParams.value = kMemsetVal1;
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(graphExec, childMemset, &memsetParams));
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graphExec, childMemcpy, C_d, S_d, Nbytes,
                                                hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    for (size_t i = 0; i < N; ++i) expected[i] = kMemsetVal1 + S_h[i];
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  SECTION("Queries on the moved graph still see its nodes") {
    size_t numNodes = 0;
    HIP_CHECK(hipGraphGetNodes(childGraph, nullptr, &numNodes));
    REQUIRE(numNodes == 3);
  }

  SECTION("Cloning the moved graph copies the adopted nodes") {
    // The handle resolves to the nodes the parent adopted, so the clone is a copy of them
    // rather than of the empty shell the caller is left holding.
    hipGraph_t clonedGraph = nullptr;
    HIP_CHECK(hipGraphClone(&clonedGraph, childGraph));
    size_t numNodes = 0;
    HIP_CHECK(hipGraphGetNodes(clonedGraph, nullptr, &numNodes));
    REQUIRE(numNodes == 3);
    // The clone is independent: it is not owned by the parent and destroys on its own.
    HIP_CHECK(hipGraphDestroy(clonedGraph));
  }

  SECTION("The moved graph cannot be used independently") {
    // What matters here is that each operation is rejected. The two runtimes disagree on
    // which code to use: AMD reports hipErrorInvalidValue throughout, while CUDA reports
    // hipErrorNotSupported for instantiation and re-parenting (and hipErrorInvalidValue
    // for destroy and exec update), so those cases accept either.

    // Owned by the parent now: destroyed with it, not on its own.
    HIP_CHECK_ERROR(hipGraphDestroy(childGraph), hipErrorInvalidValue);

    hipGraphExec_t strayExec = nullptr;
    HIP_CHECK_ERRORS(hipGraphInstantiate(&strayExec, childGraph, nullptr, nullptr, 0),
                     hipErrorInvalidValue, hipErrorNotSupported);
    HIP_CHECK_ERRORS(hipGraphInstantiateWithFlags(&strayExec, childGraph, 0),
                     hipErrorInvalidValue, hipErrorNotSupported);

    hipGraph_t otherParent = nullptr;
    HIP_CHECK(hipGraphCreate(&otherParent, 0));
    hipGraphNode_t strayNode = nullptr;
    HIP_CHECK_ERRORS(hipGraphAddChildGraphNode(&strayNode, otherParent, nullptr, 0, childGraph),
                     hipErrorInvalidValue, hipErrorNotSupported);
    hipGraphNodeParams strayParams{};
    strayParams.type = hipGraphNodeTypeGraph;
    strayParams.graph.graph = childGraph;
    strayParams.graph.ownership = hipGraphChildGraphOwnershipMove;
    HIP_CHECK_ERRORS(hipGraphAddNode(&strayNode, otherParent, nullptr, 0, &strayParams),
                     hipErrorInvalidValue, hipErrorNotSupported);
    HIP_CHECK(hipGraphDestroy(otherParent));

    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult updateResult;
    HIP_CHECK_ERROR(hipGraphExecUpdate(graphExec, childGraph, &errorNode, &updateResult),
                    hipErrorInvalidValue);

    // None of the rejections disturbed the graph.
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    HipTest::checkTest(expected.data(), result_h.data(), N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  // Destroys the moved child graph too; childGraph must not be destroyed separately.
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(S_d));
}
#endif  // HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12090)

/*
   Ownership MOVE hands the parent the caller's graph object itself, so it is legal only
   for a standalone graph that no node already owns, and only when that graph does not
   already contain the destination. Both rejected shapes would otherwise corrupt the tree:

     - Moving a node's embedded graph (what hipGraphChildGraphNodeGetGraph returns for a
       cloned child) would give one node two owners, so destroying both parents frees it
       twice.
     - Moving a graph into something it already contains would make it transitively
       contain itself, recursing without bound when instantiate clones the tree.
 */
#if HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12090)
HIP_TEST_CASE(Unit_hipGraphExecChildGraphNodeSetParams_MovedChildGraphOwnershipNegative) {
  SECTION("Move of a graph already embedded in a cloned child node is rejected") {
    hipGraph_t childGraph, parentA, parentB;
    HIP_CHECK(hipGraphCreate(&childGraph, 0));
    HIP_CHECK(hipGraphCreate(&parentA, 0));
    HIP_CHECK(hipGraphCreate(&parentB, 0));

    hipGraphNode_t childEmpty;
    HIP_CHECK(hipGraphAddEmptyNode(&childEmpty, childGraph, nullptr, 0));

    hipGraphNode_t clonedChildNode;
    HIP_CHECK(hipGraphAddChildGraphNode(&clonedChildNode, parentA, nullptr, 0, childGraph));

    // For a cloned child this is the node's own graph, not the caller's handle.
    hipGraph_t embeddedGraph = nullptr;
    HIP_CHECK(hipGraphChildGraphNodeGetGraph(clonedChildNode, &embeddedGraph));
    REQUIRE(embeddedGraph != nullptr);
    REQUIRE(embeddedGraph != childGraph);

    hipGraphNodeParams moveParams{};
    moveParams.type = hipGraphNodeTypeGraph;
    moveParams.graph.graph = embeddedGraph;
    moveParams.graph.ownership = hipGraphChildGraphOwnershipMove;

    hipGraphNode_t movedNode;
    HIP_CHECK_ERRORS(hipGraphAddNode(&movedNode, parentB, nullptr, 0, &moveParams),
                    hipErrorInvalidValue, hipErrorNotSupported);

    // The rejection left ownership untouched: parentA still owns the embedded node, so
    // tearing both parents down must not free it twice.
    HIP_CHECK(hipGraphDestroy(parentB));
    HIP_CHECK(hipGraphDestroy(parentA));
    HIP_CHECK(hipGraphDestroy(childGraph));
  }

  // Both of these sections segfault on cuda at hipGraphAddNode call.
  #if HT_AMD
  SECTION("Move of a graph into its own descendant is rejected") {
    hipGraph_t outer, inner;
    HIP_CHECK(hipGraphCreate(&outer, 0));
    HIP_CHECK(hipGraphCreate(&inner, 0));

    hipGraphNode_t innerEmpty;
    HIP_CHECK(hipGraphAddEmptyNode(&innerEmpty, inner, nullptr, 0));

    hipGraphNode_t childNode;
    HIP_CHECK(hipGraphAddChildGraphNode(&childNode, outer, nullptr, 0, inner));

    hipGraph_t embeddedInner = nullptr;
    HIP_CHECK(hipGraphChildGraphNodeGetGraph(childNode, &embeddedInner));
    REQUIRE(embeddedInner != nullptr);

    // outer already contains embeddedInner, so moving outer underneath it would make
    // outer contain itself.
    hipGraphNodeParams moveParams{};
    moveParams.type = hipGraphNodeTypeGraph;
    moveParams.graph.graph = outer;
    moveParams.graph.ownership = hipGraphChildGraphOwnershipMove;

    hipGraphNode_t cycleNode;
    HIP_CHECK_ERROR(hipGraphAddNode(&cycleNode, embeddedInner, nullptr, 0, &moveParams),
                    hipErrorInvalidValue);

    // outer is still instantiable, so the rejected call did not disturb it.
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, outer, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphExecDestroy(graphExec));

    HIP_CHECK(hipGraphDestroy(outer));
    HIP_CHECK(hipGraphDestroy(inner));
  }

  SECTION("Move of a graph into itself is rejected") {
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t emptyNode;
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));

    hipGraphNodeParams moveParams{};
    moveParams.type = hipGraphNodeTypeGraph;
    moveParams.graph.graph = graph;
    moveParams.graph.ownership = hipGraphChildGraphOwnershipMove;

    hipGraphNode_t selfNode;
    HIP_CHECK_ERROR(hipGraphAddNode(&selfNode, graph, nullptr, 0, &moveParams),
                    hipErrorInvalidValue);

    HIP_CHECK(hipGraphDestroy(graph));
  }
  #endif  // HT_AMD
}
#endif  // HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12090)
