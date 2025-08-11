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

/**
Testcase Scenarios of hipGraphAddChildGraphNode API:

Functional:
1. Create child graph as root node and execute the main graph.
2. Create multiple child graph nodes and check the behaviour.
3. Clone the child graph node, Add new nodes and execute the cloned graph.
4. Create child graph, add it to main graph and execute child graph.
5. Pass original graph as child graph and execute the org graph.
6. This test case verifies nested graph functionality. Parent graph
   containing child graph, which in turn, contains another child graph.
   Execute the graph in loop taking random input data and Validate the
   output in each iteration.
7. This test case verifies clones the nested graph created in scenario6.
   Execute the cloned graph in loop taking random input data and Validate
   the output in each iteration.
8. Verify if an empty graph can be added as child node.
9. Create the nested graph of scenario6 and update the property of add kernel
   node (innermost graph) with subtract kernel functionality. Clone the graph.
   Execute both the updated graph.
10. The updated nested graph in 9 is cloned and the cloned graph is then
    executed and the result is validated.
11. Create the nested graph of 6 and update the block size and grid size
    property of add kernel node.
12. Create the nested graph of 6 and delete the add kernel node
    (innermost graph) and add a subtract kernel node.
13. The updated nested graph in 12 is cloned and the cloned graph is then
    executed and the result is validated.
14. Create the nested graph of 6 and delete the add kernel node
    (innermost graph), add a child graph that contains an event record node,
    a subtract kernel node followed by another event record node. Clone the
    graph. Execute both the original and cloned graph.
15. The updated nested graph in 14 is cloned and the cloned graph is then
    executed and the result is validated.
16. Create one nested graph per GPU context. Execute all the created graphs
    in their respective GPUs and validate the output.
17. Functional Test to use child node as barrier to wait for multiple nodes.
    This test uses child nodes to resolve dependencies between graphs. 4
    graphs are created. Graph1 contains 3 independent memcpy h2d nodes, graph2
    contains 3 independent kernel nodes and graph3 contains 3 independent
    memcpy d2h nodes. Graph1, graph2 and graph3 are added as child nodes in
    graph4. Graph4 is validated for functionality.

Negative:
1. Pass nullptr to graph node
2. Pass nullptr to graph
3. Pass invalid number of numDepdencies
4. Pass nullptr to child graph
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define TEST_LOOP_SIZE 50
/*
This testcase verifies the negative scenarios of
hipGraphAddChildGraphNode API
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1;
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr,
      &A_h, &B_h, nullptr,
      N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, childGraphNode1;
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
        0, A_h, B_d,
        Nbytes, hipMemcpyDeviceToHost));

  SECTION("Pass nullptr to graph noe") {
    REQUIRE(hipGraphAddChildGraphNode(nullptr, graph,
          nullptr, 0, childgraph1)
        == hipErrorInvalidValue);
  }

  SECTION("Pass nullptr to graph") {
    REQUIRE(hipGraphAddChildGraphNode(&childGraphNode1, nullptr,
          nullptr, 0, childgraph1)
        == hipErrorInvalidValue);
  }

  SECTION("Pass nullptr to child graph") {
    REQUIRE(hipGraphAddChildGraphNode(&childGraphNode1, graph,
          nullptr, 0, nullptr)
        == hipErrorInvalidValue);
  }

  SECTION("Pass invalid depdencies") {
    REQUIRE(hipGraphAddChildGraphNode(&childGraphNode1, graph,
          nullptr, 10, childgraph1)
        == hipErrorInvalidValue);
  }
}

/*
This testcase verifies the following scenario
Creates the graph, add the graph as a child node
and verify the number of the nodes in the original graph
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h, &B_h, nullptr, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1;
  size_t numNodes;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, graph));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &memcpyH2D_A, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify number of nodes
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(numNodes == 3);
  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the following scenario
Create graph, Add child nodes to the graph and execute only the
child graph node and verify the behaviour
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_ExecuteChildGraph") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1;
  hipGraphExec_t graphExec;
  int *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(nullptr, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1, memcpyH2D_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, nullptr,
                                    0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
                                    0, A_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, C_d, C_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, A_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));

  HIP_CHECK(hipGraphAddDependencies(childgraph1, &memcpyH2D_B,
                                    &memcpyH2D_A, 1));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, childgraph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != A_h[i]) {
      INFO("Validation failed B_h[i] " <<  B_h[i]  << "A_h[i] "<< A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(nullptr, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the following scenario
creates graph, Add child nodes to graph, clone the graph and execute
the cloned graph
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_CloneChildGraph") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1, clonedgraph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h, &B_h, nullptr, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&clonedgraph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));

  // Added new memcpy node to the cloned graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_h, A_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyH2D_B, 1));

  // Cloned the graph
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != A_h[i]) {
      INFO("Validation failed B_h[i] " <<  B_h[i]  << "A_h[i] "<< A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the following scenario
Create graph, add multiple child nodes and validates the
behaviour
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_MultipleChildNodes") {
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

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1,
                 childGraphNode2, memcpyD2H_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph2, nullptr,
                                    0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode2, graph,
                                      nullptr, 0, childgraph2));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr,
                                    0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0,
                                                        &kernelNodeParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1,
                                    &childGraphNode2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode2,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
/**
 This testcase verifies hipGraphAddChildGraphNode functionality
 where root node is the child node.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_SingleChildNode") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  size_t NElem{N};
  int memsetVal{};
  hipGraph_t childgraph;
  hipGraphNode_t ChildGraphNode;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&childgraph, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, childgraph, nullptr, 0,
                                                              &memsetParams));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, childgraph, nullptr, 0,
                                                              &memsetParams));

  void* kernelArgs1[] = {&C_d, &memsetVal, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(HipTest::memsetReverse<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, childgraph, nullptr, 0,
                                                        &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph, nullptr,
                                    0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, childgraph, nullptr,
                                    0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, childgraph, nullptr, 0,
                                                        &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_A,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_B,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memsetKer_C,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &kernel_vecAdd,
                                    &memcpyD2H_C, 1));

  HIP_CHECK(hipGraphAddChildGraphNode(&ChildGraphNode, graph,
                                      nullptr, 0, childgraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

// Kernel functions
static __global__ void ker_vec_mul(int *A, int *B, int *C) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  C[i] = A[i]*B[i];
}

static __global__ void ker_vec_add(int *A, int *B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  A[i] = A[i] + B[i];
}

static __global__ void ker_vec_sub(int *A, int *B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  A[i] = A[i] - B[i];
}

static __global__ void ker_vec_sqr(int *A, int *B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  A[i] = B[i]*B[i];
}

enum class updateGraphNodeTests {
  normalTest,
  updateFunKerNodParamTest,
  updateGrdBlkParamTest,
  deleteAddNewKerNodTest,
  addAnotherChildNodeTest
};

/**
 Internal class for creating nested graphs.
 */
typedef class nestedGraph {
  const int const_val1 = 11;
  const int const_val2 = 7;
  const int N = 1024;
  size_t Nbytes;
  const int threadsPerBlock = 256;
  const int blocks = (N/threadsPerBlock);
  const int threadsPerBlockUpd = 128;
  const int blocksUpd = (N/threadsPerBlockUpd);
  hipGraphNode_t memset_B1, memset_B2;
  hipGraphNode_t memcpyH2D_A1, memcpyH2D_A2, memcpyD2H_A3;
  hipGraphNode_t vec_mul1, vec_mul2, vec_add, vec_sqr, vec_sub;
  hipGraphNode_t child_node1, child_node2, child_node3;
  hipGraph_t graph[4];  // 4 level graph
  hipKernelNodeParams kerNodeParams1{}, kerNodeParams2{},
  kerNodeParams3{}, kerNodeParams4{};
  int *A1_d, *A2_d, *A1_h, *A2_h, *A3_h;
  int *B1_d, *B2_d, *C1_d, *C2_d;
  hipMemsetParams memsetParams{};
  hipEvent_t eventstart, eventend;
  hipGraphNode_t event_start, event_final;

 public:
  // Create a nested Graph
  nestedGraph() {
    Nbytes = N * sizeof(int);
    // Allocate device buffers
    HIP_CHECK(hipMalloc(&A1_d, Nbytes));
    HIP_CHECK(hipMalloc(&A2_d, Nbytes));
    HIP_CHECK(hipMalloc(&B1_d, Nbytes));
    HIP_CHECK(hipMalloc(&B2_d, Nbytes));
    HIP_CHECK(hipMalloc(&C1_d, Nbytes));
    HIP_CHECK(hipMalloc(&C2_d, Nbytes));
    // Allocate host buffers
    A1_h = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(A1_h != nullptr);
    A2_h = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(A2_h != nullptr);
    A3_h = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(A3_h != nullptr);
    // Create all the 3 level graphs
    HIP_CHECK(hipGraphCreate(&graph[0], 0));
    HIP_CHECK(hipGraphCreate(&graph[1], 0));
    HIP_CHECK(hipGraphCreate(&graph[2], 0));
    HIP_CHECK(hipGraphCreate(&graph[3], 0));
    // Add the nodes to lowest level graph[2]
    void* kernelArgs1[] = {&A1_d, &B1_d, &C1_d};
    kerNodeParams1.func =
                reinterpret_cast<void *>(ker_vec_mul);
    kerNodeParams1.gridDim = dim3(blocks);
    kerNodeParams1.blockDim = dim3(threadsPerBlock);
    kerNodeParams1.sharedMemBytes = 0;
    kerNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
    kerNodeParams1.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&vec_mul1, graph[2], nullptr, 0,
                                            &kerNodeParams1));
    void* kernelArgs2[] = {&A2_d, &B2_d, &C2_d};
    kerNodeParams2.func =
                reinterpret_cast<void *>(ker_vec_mul);
    kerNodeParams2.gridDim = dim3(blocks);
    kerNodeParams2.blockDim = dim3(threadsPerBlock);
    kerNodeParams2.sharedMemBytes = 0;
    kerNodeParams2.kernelParams = reinterpret_cast<void**>(kernelArgs2);
    kerNodeParams2.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&vec_mul2, graph[2], nullptr, 0,
                                            &kerNodeParams2));
    void* kernelArgs3[] = {&C1_d, &C2_d};
    kerNodeParams3.func =
                reinterpret_cast<void *>(ker_vec_add);
    kerNodeParams3.gridDim = dim3(blocks);
    kerNodeParams3.blockDim = dim3(threadsPerBlock);
    kerNodeParams3.sharedMemBytes = 0;
    kerNodeParams3.kernelParams = reinterpret_cast<void**>(kernelArgs3);
    kerNodeParams3.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&vec_add, graph[2], nullptr, 0,
                                            &kerNodeParams3));
    // Resolve Dependencies in graph[2]
    HIP_CHECK(hipGraphAddDependencies(graph[2], &vec_mul1, &vec_add, 1));
    HIP_CHECK(hipGraphAddDependencies(graph[2], &vec_mul2, &vec_add, 1));
    // Add nodes to graph[1]
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(B1_d);
    memsetParams.value = const_val1;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(int);
    memsetParams.width = N;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memset_B1, graph[1], nullptr, 0,
                                                    &memsetParams));
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(B2_d);
    memsetParams.value = const_val2;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(int);
    memsetParams.width = N;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memset_B2, graph[1], nullptr, 0,
                                                    &memsetParams));
    HIP_CHECK(hipGraphAddChildGraphNode(&child_node1, graph[1],
                                      nullptr, 0, graph[2]));
    void* kernelArgs4[] = {&C1_d, &C1_d};
    kerNodeParams3.func =
                reinterpret_cast<void *>(ker_vec_sqr);
    kerNodeParams3.gridDim = dim3(blocks);
    kerNodeParams3.blockDim = dim3(threadsPerBlock);
    kerNodeParams3.sharedMemBytes = 0;
    kerNodeParams3.kernelParams = reinterpret_cast<void**>(kernelArgs4);
    kerNodeParams3.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&vec_sqr, graph[1], nullptr, 0,
                                            &kerNodeParams3));
    HIP_CHECK(hipGraphAddDependencies(graph[1], &memset_B1, &child_node1, 1));
    HIP_CHECK(hipGraphAddDependencies(graph[1], &memset_B2, &child_node1, 1));
    HIP_CHECK(hipGraphAddDependencies(graph[1], &child_node1, &vec_sqr, 1));
    // Add nodes to graph[0]
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A1, graph[0], nullptr,
                            0, A1_d, A1_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A2, graph[0], nullptr,
                            0, A2_d, A2_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A3, graph[0], nullptr,
                            0, A3_h, C1_d, Nbytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphAddChildGraphNode(&child_node2, graph[0],
                                      nullptr, 0, graph[1]));
    HIP_CHECK(hipGraphAddDependencies(graph[0], &memcpyH2D_A1,
              &child_node2, 1));
    HIP_CHECK(hipGraphAddDependencies(graph[0], &memcpyH2D_A2,
              &child_node2, 1));
    HIP_CHECK(hipGraphAddDependencies(graph[0], &child_node2,
              &memcpyD2H_A3, 1));
  }
  // Fill Random Input Data
  void fillRandInpData() {
    unsigned int seed = time(nullptr);
    for (int i = 0; i < N; i++) {
      A1_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
      A2_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
    }
  }
  // Get the root graph
  hipGraph_t* getRootGraph() {
    return &graph[0];
  }
  // Get the root graph
  void updateInnermostNode(updateGraphNodeTests updatetype) {
    hipGraph_t embGraph1, embGraph2;
    // Get the embedded graph from child_node2
    HIP_CHECK(hipGraphChildGraphNodeGetGraph(child_node2, &embGraph2));
    size_t numNodes{};
    HIP_CHECK(hipGraphGetNodes(embGraph2, nullptr, &numNodes));
    hipGraphNode_t* nodes =
    reinterpret_cast<hipGraphNode_t *>(
        malloc(numNodes*sizeof(hipGraphNode_t)));
    HIP_CHECK(hipGraphGetNodes(embGraph2, nodes, &numNodes));
    // Get the Graph node from the embedded graph
    size_t nodeIdx = 0;
    for (size_t idx = 0; idx < numNodes; idx++) {
      hipGraphNodeType nodeType;
      HIP_CHECK(hipGraphNodeGetType(nodes[idx], &nodeType));
      if (nodeType == hipGraphNodeTypeGraph) {
        nodeIdx = idx;
        break;
      }
    }
    // Extract the embedded graph from the graph node
    HIP_CHECK(hipGraphChildGraphNodeGetGraph(nodes[nodeIdx], &embGraph1));
    free(nodes);
    numNodes = 0;
    HIP_CHECK(hipGraphGetNodes(embGraph1, nullptr, &numNodes));
    nodes = reinterpret_cast<hipGraphNode_t *>(
          malloc(numNodes*sizeof(hipGraphNode_t)));
    // Get the kernel node from the extracted embedded graph
    HIP_CHECK(hipGraphGetNodes(embGraph1, nodes, &numNodes));
    nodeIdx = 0;
    hipKernelNodeParams nodeParam;
    for (size_t idx = 0; idx < numNodes; idx++) {
      hipGraphNodeType nodeType;
      HIP_CHECK(hipGraphNodeGetType(nodes[idx], &nodeType));
      if (nodeType == hipGraphNodeTypeKernel) {
        HIP_CHECK(hipGraphKernelNodeGetParams(nodes[idx], &nodeParam));
        if (nodeParam.func == reinterpret_cast<void *>(ker_vec_add)) {
          nodeIdx = idx;
          break;
        }
      }
    }
    if (updatetype == updateGraphNodeTests::updateFunKerNodParamTest) {
      nodeParam.func = reinterpret_cast<void *>(ker_vec_sub);
      HIP_CHECK(hipGraphKernelNodeSetParams(nodes[nodeIdx], &nodeParam));
    } else if (updatetype == updateGraphNodeTests::deleteAddNewKerNodTest) {
      // delete the kernel add node
      HIP_CHECK(hipGraphDestroyNode(nodes[nodeIdx]));
      // add kernel subtract node to embGraph1
      void* kernelArgs[] = {&C1_d, &C2_d};
      kerNodeParams3.func =
                reinterpret_cast<void *>(ker_vec_sub);
      kerNodeParams3.gridDim = dim3(blocks);
      kerNodeParams3.blockDim = dim3(threadsPerBlock);
      kerNodeParams3.sharedMemBytes = 0;
      kerNodeParams3.kernelParams = reinterpret_cast<void**>(kernelArgs);
      kerNodeParams3.extra = nullptr;
      HIP_CHECK(hipGraphAddKernelNode(&vec_sub, embGraph1, nullptr, 0,
                                      &kerNodeParams3));
      // Create new dependencies
      for (size_t idx = 0; idx < numNodes; idx++) {
        if (idx == nodeIdx) {
          continue;
        }
        HIP_CHECK(hipGraphAddDependencies(embGraph1, &nodes[idx],
                                          &vec_sub, 1));
      }
    } else if (updatetype == updateGraphNodeTests::updateGrdBlkParamTest) {
      nodeParam.blockDim = threadsPerBlockUpd;
      nodeParam.gridDim = blocksUpd;
      HIP_CHECK(hipGraphKernelNodeSetParams(nodes[nodeIdx], &nodeParam));
    } else if (updatetype == updateGraphNodeTests::addAnotherChildNodeTest) {
      // delete the kernel add node
      HIP_CHECK(hipGraphDestroyNode(nodes[nodeIdx]));
      // add graph EventRecordNode -> Subtract Kernel -> EventRecordNode as
      // child node
      void* kernelArgs[] = {&C1_d, &C2_d};
      kerNodeParams3.func =
                reinterpret_cast<void *>(ker_vec_sub);
      kerNodeParams3.gridDim = dim3(blocks);
      kerNodeParams3.blockDim = dim3(threadsPerBlock);
      kerNodeParams3.sharedMemBytes = 0;
      kerNodeParams3.kernelParams = reinterpret_cast<void**>(kernelArgs);
      kerNodeParams3.extra = nullptr;
      HIP_CHECK(hipGraphAddKernelNode(&vec_sub, graph[3], nullptr, 0,
                                      &kerNodeParams3));
      HIP_CHECK(hipEventCreate(&eventstart));
      HIP_CHECK(hipEventCreate(&eventend));
      HIP_CHECK(hipGraphAddEventRecordNode(&event_start, graph[3], nullptr,
                                           0, eventstart));
      HIP_CHECK(hipGraphAddEventRecordNode(&event_final, graph[3], nullptr,
                                           0, eventend));
      HIP_CHECK(hipGraphAddDependencies(graph[3], &event_start,
                                        &vec_sub, 1));
      HIP_CHECK(hipGraphAddDependencies(graph[3], &vec_sub,
                                        &event_final, 1));
      HIP_CHECK(hipGraphAddChildGraphNode(&child_node3, embGraph1, nullptr,
                                        0, graph[3]));
      // Create new dependencies
      for (size_t idx = 0; idx < numNodes; idx++) {
        if (idx == nodeIdx) {
          continue;
        }
        HIP_CHECK(hipGraphAddDependencies(embGraph1, &nodes[idx],
                                          &child_node3, 1));
      }
    }
    free(nodes);
  }
  // Function to validate result
  void validateOutData(updateGraphNodeTests updatetype) {
    if ((updatetype == updateGraphNodeTests::normalTest) ||
    (updatetype == updateGraphNodeTests::updateGrdBlkParamTest)) {
      for (int i = 0; i < N; i++) {
        int result = (const_val1*A1_h[i] + const_val2*A2_h[i]);
        result = result * result;
        REQUIRE(result == A3_h[i]);
      }
    } else if ((updatetype == updateGraphNodeTests::deleteAddNewKerNodTest)
        || (updatetype == updateGraphNodeTests::updateFunKerNodParamTest)
        || (updatetype == updateGraphNodeTests::addAnotherChildNodeTest)) {
      for (int i = 0; i < N; i++) {
        int result = (const_val1*A1_h[i] - const_val2*A2_h[i]);
        result = result * result;
        REQUIRE(result == A3_h[i]);
      }
    }
  }
  // Destroy resources
  ~nestedGraph() {
    // Free all allocated buffers
    HIP_CHECK(hipFree(C2_d));
    HIP_CHECK(hipFree(C1_d));
    HIP_CHECK(hipFree(B2_d));
    HIP_CHECK(hipFree(B1_d));
    HIP_CHECK(hipFree(A2_d));
    HIP_CHECK(hipFree(A1_d));
    free(A3_h);
    free(A2_h);
    free(A1_h);
    HIP_CHECK(hipGraphDestroy(graph[3]));
    HIP_CHECK(hipGraphDestroy(graph[2]));
    HIP_CHECK(hipGraphDestroy(graph[1]));
    HIP_CHECK(hipGraphDestroy(graph[0]));
  }
} clNestedGraph;

/**
 Complex Scenario: This testcase verifies nested graph functionality.
 Parent graph containing child graph, which in turn, contains another
 child graph.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_Cmplx_NestedGraphs") {
  hipGraph_t *graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, (*graph), nullptr,
                                nullptr, 0));
  for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
    nestedGraphObj.fillRandInpData();
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    nestedGraphObj.validateOutData(updateGraphNodeTests::normalTest);
  }
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
}

/**
 Complex Scenario: This testcase verifies cloned nested graph functionality.
 Clone the nested graph and execute the clone graph.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxClone_NestedGraphs") {
  hipGraph_t *graph, clonedGraph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  HIP_CHECK(hipGraphClone(&clonedGraph, *graph));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedGraph, nullptr,
                                nullptr, 0));
  for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
    nestedGraphObj.fillRandInpData();
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    nestedGraphObj.validateOutData(updateGraphNodeTests::normalTest);
  }
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 Scenario: Adding an empty graph to Child Graph Node.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_EmptyGraphAsChildNode") {
  hipGraph_t graph, graphChild;
  hipGraphNode_t child_node;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&graphChild, 0));
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, graph,
                                    nullptr, 0, graphChild));
  HIP_CHECK(hipGraphDestroy(graphChild));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 Complex Scenario: This testcase verifies the behavior of a nested graph
 when one of the child graph node is updated. In this test the kernel node
 function is updated to a different function.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_UpdKerFun") {
  hipGraph_t *graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  nestedGraphObj.updateInnermostNode(
        updateGraphNodeTests::updateFunKerNodParamTest);
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, (*graph), nullptr,
                                nullptr, 0));
  nestedGraphObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  nestedGraphObj.validateOutData(
                updateGraphNodeTests::updateFunKerNodParamTest);
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
}

/**
 Complex Scenario: This testcase verifies the behavior of a nested graph
 when one of the child graph node is updated. In this test the kernel node
 function is updated to a different function and the nested graph is cloned.
 Execute the cloned graph and validate the results.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_UpdKerFun_Clone") {
  hipGraph_t *graph, clonedGraph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  nestedGraphObj.updateInnermostNode(
        updateGraphNodeTests::updateFunKerNodParamTest);
  HIP_CHECK(hipGraphClone(&clonedGraph, *graph));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedGraph, nullptr,
                                nullptr, 0));
  nestedGraphObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  nestedGraphObj.validateOutData(
                updateGraphNodeTests::updateFunKerNodParamTest);
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 Complex Scenario: This testcase verifies the behavior of a nested graph
 when one of the child graph node is updated. In this test the kernel node
 parameters - blocksize and gridsize are updated.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_UpdKerDim") {
  hipGraph_t *graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  nestedGraphObj.updateInnermostNode(
        updateGraphNodeTests::updateGrdBlkParamTest);
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, (*graph), nullptr,
                                nullptr, 0));
  nestedGraphObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  nestedGraphObj.validateOutData(
                updateGraphNodeTests::updateGrdBlkParamTest);
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
}

/**
 Complex Scenario: This testcase verifies the behavior of a nested graph
 when one of the nodes inside a child graph node is deleted and replaced with
 a new node.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_DelAddNode") {
  hipGraph_t *graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  nestedGraphObj.updateInnermostNode(
        updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, (*graph), nullptr,
                                nullptr, 0));
  nestedGraphObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  nestedGraphObj.validateOutData(
                updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
}

/**
 Complex Scenario: This testcase verifies the behavior of a cloned nested
 graph when one of the nodes inside a child graph node is deleted and
 replaced with a new node. After modifying the original graph it is cloned
 and the cloned graph is executed and validated.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_AddNode_Clone") {
  hipGraph_t *graph, clonedGraph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  nestedGraphObj.updateInnermostNode(
        updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipGraphClone(&clonedGraph, *graph));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedGraph, nullptr,
                                nullptr, 0));
  nestedGraphObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  nestedGraphObj.validateOutData(
                updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 Complex Scenario: This testcase verifies the behavior of a nested graph
 when one of the nodes inside a child graph node is deleted and replaced with
 a new child graph node.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_AddChdNode") {
  hipGraph_t *graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  nestedGraphObj.updateInnermostNode(
        updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, (*graph), nullptr,
                                nullptr, 0));
  nestedGraphObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  nestedGraphObj.validateOutData(
                updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
}

/**
 Complex Scenario: This testcase verifies the behavior of a cloned nested graph
 when one of the nodes inside a child graph node is deleted and replaced with
 a new child graph node.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_AddChdNode_Clone") {
  hipGraph_t *graph, clonedGraph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  class nestedGraph nestedGraphObj;
  graph = nestedGraphObj.getRootGraph();
  nestedGraphObj.updateInnermostNode(
        updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipGraphClone(&clonedGraph, *graph));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedGraph, nullptr,
                                nullptr, 0));
  nestedGraphObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  nestedGraphObj.validateOutData(
                updateGraphNodeTests::deleteAddNewKerNodTest);
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

// Function to validate result
static void validateResults(int *A1_h, int *A2_h, size_t N) {
  for (size_t i = 0; i < N; i++) {
    int result = (A1_h[i]*A1_h[i]);
    REQUIRE(result == A2_h[i]);
  }
}

/**
 Functional Test to use child node as barrier to wait for multiple nodes.
 This test uses child nodes to resolve dependencies between graphs. 4
 graphs are created. Graph1 contains 3 independent memcpy h2d nodes, graph2
 contains 3 independent kernel nodes and graph3 contains 3 independent
 memcpy d2h nodes. Graph1, graph2 and graph3 are added as child nodes in
 graph4. Graph4 is validated for functionality.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_MultGraphsAsSingleGraph") {
  size_t size = 1024;
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                            threadsPerBlock, size);
  hipGraph_t graph1, graph2, graph3, graph4;
  std::vector<hipGraphNode_t> nodeDependencies;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphCreate(&graph3, 0));
  HIP_CHECK(hipGraphCreate(&graph4, 0));
  int *inputVec_d1{nullptr}, *inputVec_h1{nullptr}, *outputVec_h1{nullptr},
      *outputVec_d1{nullptr};
  int *inputVec_d2{nullptr}, *inputVec_h2{nullptr}, *outputVec_h2{nullptr},
      *outputVec_d2{nullptr};
  int *inputVec_d3{nullptr}, *inputVec_h3{nullptr}, *outputVec_h3{nullptr},
      *outputVec_d3{nullptr};
  // host and device allocation
  HipTest::initArrays<int>(&inputVec_d1, &outputVec_d1, nullptr,
               &inputVec_h1, &outputVec_h1, nullptr, size, false);
  HipTest::initArrays<int>(&inputVec_d2, &outputVec_d2, nullptr,
               &inputVec_h2, &outputVec_h2, nullptr, size, false);
  HipTest::initArrays<int>(&inputVec_d3, &outputVec_d3, nullptr,
               &inputVec_h3, &outputVec_h3, nullptr, size, false);
  // add nodes to graph
  hipGraphNode_t memcpyH2D_1, memcpyH2D_2, memcpyH2D_3;
  hipGraphNode_t vecSqr1, vecSqr2, vecSqr3;
  hipGraphNode_t memcpyD2H_1, memcpyD2H_2, memcpyD2H_3;
  hipGraphNode_t childGraphNode1, childGraphNode2, childGraphNode3;
  // Create memcpy h2d nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_1, graph1, nullptr,
  0, inputVec_d1, inputVec_h1, (sizeof(int)*size), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_2, graph1, nullptr,
  0, inputVec_d2, inputVec_h2, (sizeof(int)*size), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_3, graph1, nullptr,
  0, inputVec_d3, inputVec_h3, (sizeof(int)*size), hipMemcpyHostToDevice));
  // Create child node and add it to graph4
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph4, nullptr, 0,
            graph1));
  nodeDependencies.clear();
  nodeDependencies.push_back(childGraphNode1);
  // Creating kernel nodes
  hipKernelNodeParams kerNodeParams1{}, kerNodeParams2{}, kerNodeParams3{};
  void* kernelArgs1[] = {reinterpret_cast<void*>(&inputVec_d1),
                        reinterpret_cast<void*>(&outputVec_d1),
                        reinterpret_cast<void*>(&size)};
  kerNodeParams1.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams1.gridDim = dim3(blocks);
  kerNodeParams1.blockDim = dim3(threadsPerBlock);
  kerNodeParams1.sharedMemBytes = 0;
  kerNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kerNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&vecSqr1, graph2, nullptr, 0,
            &kerNodeParams1));
  void* kernelArgs2[] = {reinterpret_cast<void*>(&inputVec_d2),
                        reinterpret_cast<void*>(&outputVec_d2),
                        reinterpret_cast<void*>(&size)};
  kerNodeParams2.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams2.gridDim = dim3(blocks);
  kerNodeParams2.blockDim = dim3(threadsPerBlock);
  kerNodeParams2.sharedMemBytes = 0;
  kerNodeParams2.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kerNodeParams2.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&vecSqr2, graph2, nullptr, 0,
            &kerNodeParams2));
  void* kernelArgs3[] = {reinterpret_cast<void*>(&inputVec_d3),
                        reinterpret_cast<void*>(&outputVec_d3),
                        reinterpret_cast<void*>(&size)};
  kerNodeParams3.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams3.gridDim = dim3(blocks);
  kerNodeParams3.blockDim = dim3(threadsPerBlock);
  kerNodeParams3.sharedMemBytes = 0;
  kerNodeParams3.kernelParams = reinterpret_cast<void**>(kernelArgs3);
  kerNodeParams3.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&vecSqr3, graph2, nullptr, 0,
            &kerNodeParams3));
  // Create child node and add it to graph4
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode2, graph4,
            nodeDependencies.data(), nodeDependencies.size(), graph2));
  nodeDependencies.clear();
  nodeDependencies.push_back(childGraphNode2);
  // Create memcpy d2h nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_1, graph3, nullptr, 0,
  outputVec_h1, outputVec_d1, (sizeof(int)*size), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_2, graph3, nullptr, 0,
  outputVec_h2, outputVec_d2, (sizeof(int)*size), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_3, graph3, nullptr, 0,
  outputVec_h3, outputVec_d3, (sizeof(int)*size), hipMemcpyDeviceToHost));
  // Create child node and add it to graph4
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode3, graph4,
            nodeDependencies.data(), nodeDependencies.size(), graph3));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph4, nullptr,
                                nullptr, 0));
  // Execute graph
  for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
    // Inititalize random input data
    unsigned int seed = time(nullptr);
    for (size_t i = 0; i < size; i++) {
      inputVec_h1[i] = (HipTest::RAND_R(&seed) & 0xFF);
      inputVec_h2[i] = (HipTest::RAND_R(&seed) & 0xFF);
      inputVec_h3[i] = (HipTest::RAND_R(&seed) & 0xFF);
    }
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    validateResults(inputVec_h1, outputVec_h1, size);
    validateResults(inputVec_h2, outputVec_h2, size);
    validateResults(inputVec_h3, outputVec_h3, size);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  // Free
  HipTest::freeArrays<int>(inputVec_d1, outputVec_d1, nullptr,
                   inputVec_h1, outputVec_h1, nullptr, false);
  HipTest::freeArrays<int>(inputVec_d2, outputVec_d2, nullptr,
                   inputVec_h2, outputVec_h2, nullptr, false);
  HipTest::freeArrays<int>(inputVec_d3, outputVec_d3, nullptr,
                   inputVec_h3, outputVec_h3, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph4));
  HIP_CHECK(hipGraphDestroy(graph3));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/**
 Complex Scenario: This testcase verifies the behavior of a nested graph
 in multi GPU environment. Create one nested graph per GPU context. Execute
 all the created graphs in their respective GPUs and validate the output.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_CmplxNstGrph_MultGPU") {
  int devcount = 0;
  HIP_CHECK(hipGetDeviceCount(&devcount));
  // If only single GPU is detected then return
  if (devcount < 2) {
    SUCCEED("skipping the testcases as numDevices < 2");
    return;
  }
  hipGraph_t **graph = new hipGraph_t *[devcount]();
  REQUIRE(graph != nullptr);
  hipStream_t *streamForGraph = new hipStream_t[devcount];
  REQUIRE(streamForGraph != nullptr);
  hipGraphExec_t *graphExec = new hipGraphExec_t[devcount];
  REQUIRE(graphExec != nullptr);
  clNestedGraph** nestedGraphObj = new clNestedGraph *[devcount]();
  REQUIRE(nestedGraphObj != nullptr);
  // Create graph resources for each devices
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    nestedGraphObj[dev] = new clNestedGraph();
    REQUIRE(nestedGraphObj[dev] != nullptr);
    graph[dev] = nestedGraphObj[dev]->getRootGraph();
    HIP_CHECK(hipStreamCreate(&streamForGraph[dev]));
    // Instantiate and launch the childgraph
    HIP_CHECK(hipGraphInstantiate(&graphExec[dev], *(graph[dev]), nullptr,
                                 nullptr, 0));
  }
  // Execute graph in each GPU
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    nestedGraphObj[dev]->fillRandInpData();
    HIP_CHECK(hipGraphLaunch(graphExec[dev], streamForGraph[dev]));
  }
  // Wait for each device to complete task and validate the results
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streamForGraph[dev]));
    nestedGraphObj[dev]->validateOutData(
                       updateGraphNodeTests::normalTest);
  }
  // Destroy graph resources
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipStreamDestroy(streamForGraph[dev]));
    HIP_CHECK(hipGraphExecDestroy(graphExec[dev]));
    delete nestedGraphObj[dev];
  }
  delete[] nestedGraphObj;
  delete[] graphExec;
  delete[] streamForGraph;
  delete[] graph;
}
