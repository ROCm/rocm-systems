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
<<<<<<< HEAD
=======

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
>>>>>>> ae26d57c1a6 (add testcase for MovedChildGraphOwnership)
 **/

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
