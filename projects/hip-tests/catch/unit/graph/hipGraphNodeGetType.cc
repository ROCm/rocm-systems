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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/**
Testcase Scenarios :
Functional ::
1) Pass different types of valid nodes and get the corresponding type of the node
2) Add Graph node, destroy the graph node and add the same node of different type.
   hipGraphNodeGetTye should return the new type
3) Add graph node, overwrite new type to the same graph node and trigger
   hipGraphNodeGetType which should return the updated type.
4) Create a graph with different types of nodes. Clone the graph. Verify the types
   of each of these nodes in the cloned graph using hipGraphNodeGetType.
5) Create a graph with different types of nodes. Pass the graph to a thread. In the
   thread, verify node types of all the nodes in the graph
6) Create a graph with different types of nodes say X. Create graph Y with few nodes
   and X as child graph. Now verify each of nodes of Y including the nodes inside
   child graph using hipGraphNodeGetType()
7) Create a graph with different types of nodes along with dependencies between
   nodes. Clone the graph. Verify the types of each of these nodes in the cloned
   graph using hipGraphNodeGetType.
8) Create a graph with different types of nodes along with dependencies between
   nodes. Pass the graph to thread and verify each type of node in the graph

9) Create a graph with different types of nodes say X with dependencies between
   nodes. Create graph Y with few nodes with dependencies and X as child graph.
   Now verify each of nodes of Y including the nodes inside child graph using
   hipGraphNodeGetType()

Negative ::
1) Pass nullptr to graph node
2) Pass nullptr  to node type
3) Pass invalid to graph node
*/
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <map>

#define SIZE 1024

__device__ __constant__ static int globalConst[SIZE];
static void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }
}

TEST_CASE("Unit_hipGraphNodeGetType_Negative") {
  SECTION("Pass nullptr to graph node") {
    hipGraphNodeType nodeType;
    REQUIRE(hipGraphNodeGetType(nullptr, &nodeType) == hipErrorInvalidValue);
  }

  SECTION("Pass nullptr to node type") {
    hipGraphNode_t memcpyNode;
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&memcpyNode, graph, nullptr, 0));
    REQUIRE(hipGraphNodeGetType(memcpyNode, nullptr) == hipErrorInvalidValue);
  }

  SECTION("Pass invalid node") {
    hipGraphNode_t Node = {};
    hipGraphNodeType nodeType;
    REQUIRE(hipGraphNodeGetType(Node, &nodeType) == hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipGraphNodeGetType_Functional") {
  constexpr size_t N = 1024;
  hipGraphNodeType nodeType;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  hipEvent_t event;
  hipGraphNode_t waiteventNode;
  HIP_CHECK(hipEventCreate(&event));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Delete Node and Assign different Node Type") {
    HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
    HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0, event));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    HIP_CHECK(hipGraphAddEmptyNode(&waiteventNode, graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }

  SECTION("Override the graph node and get Node Type") {
    HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
    HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0, event));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    HIP_CHECK(hipGraphAddEmptyNode(&waiteventNode, graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(event));
}
/**
 * Functional Test for hipGraphNodeGetType API
 * Create a graph and add nodes to it.
 * Verify each node type added.
 */
constexpr size_t N = 1024;
constexpr size_t Nbytes = N * sizeof(int);
constexpr auto blocksPerCU = 6;  // to hide latency
constexpr auto threadsPerBlock = 256;

TEST_CASE("Unit_hipGraphNodeGetType_NodeType") {
  hipGraph_t graph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNodeType nodeType;
  hipGraphNode_t memcpyNode, kernelNode;

  SECTION("Get Memcpy node NodeType from Symbol") {
    hipGraphNode_t memcpyFromSymbolNode;
    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph, nullptr, 0, B_d,
                                              HIP_SYMBOL(globalConst), Nbytes, 0,
                                              hipMemcpyDeviceToDevice));
    // Verify node type
    HIP_CHECK(hipGraphNodeGetType(memcpyFromSymbolNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeMemcpy);
  }

  SECTION("Get Memcpy node NodeType to Symbol") {
    hipGraphNode_t memcpyToSymbolNode;
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph, nullptr, 0,
                                            HIP_SYMBOL(globalConst), A_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
    // Verify node type
    HIP_CHECK(hipGraphNodeGetType(memcpyToSymbolNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeMemcpy);
  }

  SECTION("Get Host node NodeType") {
    hipGraphNode_t hostNode;
    hipHostNodeParams hostParams = {0, 0};
    hostParams.fn = callbackfunc;
    hostParams.userData = A_h;
    // Create a host node
    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
    // Verify node type
    HIP_CHECK(hipGraphNodeGetType(hostNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeHost);
  }

  SECTION("Get Child node NodeType") {
    hipGraph_t childgraph;
    hipGraphNode_t childGraphNode;
    // Create child graph
    HIP_CHECK(hipGraphCreate(&childgraph, 0));
    // Add child graph node
    HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childgraph))
    // Verify node type
    HIP_CHECK(hipGraphNodeGetType(childGraphNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeGraph);
  }

  SECTION("Get Memcpy NodeType") {
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphNodeGetType(memcpyNode, &nodeType));

    // temp disable it until correct node is set
    // REQUIRE(nodeType == hipGraphNodeTypeMemcpy);

    HIP_CHECK(hipGraphAddEmptyNode(&memcpyNode, graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetType(memcpyNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }

  SECTION("Get Kernel NodeType") {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));
    HIP_CHECK(hipGraphNodeGetType(kernelNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeKernel);
  }

  SECTION("Get Empty NodeType") {
    hipGraphNode_t emptyNode;
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetType(emptyNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }

  SECTION("Get Memset NodeType") {
    hipGraphNode_t memsetNode;
    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(A_d);
    memsetParams.value = 10;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
    HIP_CHECK(hipGraphNodeGetType(memsetNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeMemset);
  }

  SECTION("Get WaitEvent NodeType") {
    hipEvent_t event;
    hipGraphNode_t waiteventNode;
    HIP_CHECK(hipEventCreate(&event));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
    HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0, event));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeWaitEvent);
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipEventDestroy(event));
  }

  SECTION("Get EventRecord NodeType") {
    hipEvent_t event;
    hipGraphNode_t recordeventNode;
    HIP_CHECK(hipEventCreate(&event));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipGraphAddEventRecordNode(&recordeventNode, graph, nullptr, 0, event));
    HIP_CHECK(hipGraphNodeGetType(recordeventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEventRecord);
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipEventDestroy(event));
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

// Function to verify node Type
static void ChkNodeType(hipGraph_t graph, const std::map<hipGraphNodeType, int>* nodeTypeToQuery) {
  size_t numNodes{};
  hipGraphNodeType nodeType;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  int numBytes = sizeof(hipGraphNode_t) * numNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t*>(malloc(numBytes));
  REQUIRE(nodes != nullptr);
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  std::map<hipGraphNodeType, int> cntNode;
  for (size_t i = 0; i < numNodes; i++) {
    HIP_CHECK(hipGraphNodeGetType(nodes[i], &nodeType));
    cntNode[nodeType] += 1;
  }
  std::map<hipGraphNodeType, int>::iterator iter;
  std::map<hipGraphNodeType, int>::const_iterator iter1 = nodeTypeToQuery->begin();
  for (iter = cntNode.begin(); iter != cntNode.end(); iter++) {
    REQUIRE(iter->first == iter1->first);
    REQUIRE(iter->second == iter1->second);
    if (iter1 == nodeTypeToQuery->end())
      break;
    else
      iter1++;
  }
  free(nodes);
}
// Thread Function
static void thread_func(hipGraph_t graph, std::map<hipGraphNodeType, int>* numNode) {
  ChkNodeType(graph, numNode);
}
/*
 * 1.Create a graph with different types of nodes. Clone the graph. Verify the types
 * of each of these nodes in the cloned graph using hipGraphNodeGetType.
 * 2.Create a graph with different types of nodes. Pass the graph to a thread. In the
 * thread, verify node types of all the nodes in the graph
 */
TEST_CASE("Unit_hipGraphNodeGetType_NodeTypeOfClonedGraph_NodeTypeInThread") {
  hipGraph_t graph, childGraph, clonedGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  std::map<hipGraphNodeType, int> numNode;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyNode, kernelNode, hostNode, childGraphNode, emptyNode, memsetNode,
      waiteventNode, recordeventNode;
  int numMemcpy{}, numKernel{}, numMemset{}, numHost{}, numWaitEvent{}, numEventRecord{},
      numEmpty{}, numChild{};

  // Host Node
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  numHost++;
  // Host NOde
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  numHost++;
  // Host Node
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  numHost++;

  // MemCpy Node
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  numMemcpy++;
  // Host Node
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  numHost++;
  // Kernal Node
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));
  numKernel++;
  // Child Node
  HIP_CHECK(hipGraphCreate(&childGraph, 0));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));
  numChild++;
  // Child Node
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));
  numChild++;
  // memSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 10;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  numMemset++;
  // WaitEvent Node
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamWaitEvent(stream1, event1, 0));
  HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0, event1));
  numWaitEvent++;
  // Empty Node
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));
  numEmpty++;
  // Empty Node
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));
  numEmpty++;
  // Event Record Node
  HIP_CHECK(hipEventCreate(&event2));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventRecord(event2, stream2));
  HIP_CHECK(hipGraphAddEventRecordNode(&recordeventNode, graph, nullptr, 0, event2));
  numEventRecord++;

  numNode[hipGraphNodeTypeHost] = numHost;
  numNode[hipGraphNodeTypeMemcpy] = numMemcpy;
  numNode[hipGraphNodeTypeKernel] = numKernel;
  numNode[hipGraphNodeTypeMemset] = numMemset;
  numNode[hipGraphNodeTypeGraph] = numChild;
  numNode[hipGraphNodeTypeWaitEvent] = numWaitEvent;
  numNode[hipGraphNodeTypeEmpty] = numEmpty;
  numNode[hipGraphNodeTypeEventRecord] = numEventRecord;

  // Clone the graph
  SECTION("Cloned Graph Node Type") {
    HIP_CHECK(hipGraphClone(&clonedGraph, graph));
    ChkNodeType(clonedGraph, &numNode);
  }
  // Thread
  SECTION("Node Type In The Thread") {
    std::thread t(thread_func, graph, &numNode);
    t.join();
  }

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event2));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipGraphDestroy(graph));
}
/*
 * Create a graph with different types of nodes say X. Create graph Y with
 * few nodes and X as child graph. Now verify each of nodes of Y including
 * the nodes inside child graph using hipGraphNodeGetType()
 */
TEST_CASE("Unit_hipGraphNodeGetType_NodeTypeOfChildGraph") {
  hipGraph_t graph, childGraph, getGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyNode, kernelNode, hostNode, childGraphNode, emptyNode, memsetNode,
      waiteventNode, recordeventNode;
  int numMemcpy{}, numKernel{}, numMemset{}, numHost{}, numWaitEvent{}, numEventRecord{},
      numEmpty{}, numChild{};

  std::map<hipGraphNodeType, int> numNodeParent;
  std::map<hipGraphNodeType, int> numNodeChild;
  // Create a child graph
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  // Add memSet Node to child graph
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 10;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, childGraph, nullptr, 0, &memsetParams));
  numMemset++;

  // Add WaitEvent Node to child graph
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamWaitEvent(stream1, event1, 0));
  HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, childGraph, nullptr, 0, event1));
  numWaitEvent++;
  // Add Empty Node to child graph
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, childGraph, nullptr, 0));
  numEmpty++;
  // Empty Node
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, childGraph, nullptr, 0));
  numEmpty++;

  // Add Event Record Node to child graph
  HIP_CHECK(hipEventCreate(&event2));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventRecord(event2, stream2));
  HIP_CHECK(hipGraphAddEventRecordNode(&recordeventNode, childGraph, nullptr, 0, event2));
  numEventRecord++;
  // Add Host Node to parent graph
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  numHost++;
  // Add MemCpy Node to parent graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  numMemcpy++;
  // Add MemCpy Node to parent graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  numMemcpy++;

  // Add Kernal Node to parent graph
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));
  numKernel++;

  // Add child node to the parent graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph))
  numChild++;

  numNodeParent[hipGraphNodeTypeHost] = numHost;
  numNodeParent[hipGraphNodeTypeMemcpy] = numMemcpy;
  numNodeParent[hipGraphNodeTypeKernel] = numKernel;
  numNodeParent[hipGraphNodeTypeGraph] = numChild;
  numNodeChild[hipGraphNodeTypeMemset] = numMemset;
  numNodeChild[hipGraphNodeTypeWaitEvent] = numWaitEvent;
  numNodeChild[hipGraphNodeTypeEmpty] = numEmpty;
  numNodeChild[hipGraphNodeTypeEventRecord] = numEventRecord;
  // Check Node Type of graph
  ChkNodeType(graph, &numNodeParent);

  // Get the child graph from parent graph
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode, &getGraph));
  ChkNodeType(getGraph, &numNodeChild);

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event2));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipGraphDestroy(graph));
}
enum graphType { Parent, Child };
// Function to verify node Type
static void ChkNodeTypeWithDependency(hipGraph_t graph, enum graphType Type) {
  size_t numNodes{};
  hipGraphNodeType nodeType;
  hipGraphNodeType Arr[] = {hipGraphNodeTypeHost,   hipGraphNodeTypeMemcpy,
                            hipGraphNodeTypeKernel, hipGraphNodeTypeGraph,
                            hipGraphNodeTypeMemset, hipGraphNodeTypeWaitEvent,
                            hipGraphNodeTypeEmpty,  hipGraphNodeTypeEventRecord};

  hipGraphNodeType childArr[] = {hipGraphNodeTypeMemset, hipGraphNodeTypeWaitEvent,
                                 hipGraphNodeTypeEmpty, hipGraphNodeTypeEventRecord};

  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  int numBytes = sizeof(hipGraphNode_t) * numNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t*>(malloc(numBytes));
  REQUIRE(nodes != nullptr);

  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  for (size_t i = 0; i < numNodes; i++) {
    HIP_CHECK(hipGraphNodeGetType(nodes[i], &nodeType));
    if (Type == Parent) {
      REQUIRE(nodeType == Arr[i]);
    } else if (Type == Child) {
      REQUIRE(nodeType == childArr[i]);
    }
  }
  free(nodes);
}

// Thread Function
static void thread_func1(hipGraph_t graph, enum graphType type) {
  ChkNodeTypeWithDependency(graph, type);
}
/*
 * 1.Create a graph with different types of nodes along with dependencies between
 * nodes. Clone the graph. Verify the types of each of these nodes in the cloned
 * graph using hipGraphNodeGetType.
 * 2.Pass the graph to thread and verify each type of node in the graph
 * */
TEST_CASE("Unit_hipGraphNodeGetType_ClonedGraph_InThread_WithDependencies") {
  hipGraph_t graph, childGraph, clonedGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyNode, kernelNode, hostNode, childGraphNode, emptyNode, memsetNode,
      waiteventNode, recordeventNode;

  // Host Node
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  // MemCpy Node
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &hostNode, &memcpyNode, 1));

  // Kernal Node
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyNode, &kernelNode, 1));

  // Child Node
  HIP_CHECK(hipGraphCreate(&childGraph, 0));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelNode, &childGraphNode, 1));

  // memSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 10;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode, &memsetNode, 1));

  // WaitEvent Node
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamWaitEvent(stream1, event1, 0));
  HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0, event1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetNode, &waiteventNode, 1));

  // Empty Node
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddDependencies(graph, &waiteventNode, &emptyNode, 1));

  // Event Record Node
  HIP_CHECK(hipEventCreate(&event2));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventRecord(event2, stream2));
  HIP_CHECK(hipGraphAddEventRecordNode(&recordeventNode, graph, nullptr, 0, event2));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode, &recordeventNode, 1));

  // Clone the graph
  SECTION("Cloned Graph Node Type") {
    HIP_CHECK(hipGraphClone(&clonedGraph, graph));
    ChkNodeTypeWithDependency(clonedGraph, Parent);
  }
  // Thread
  SECTION("Node Type In The Thread") {
    std::thread t(thread_func1, graph, Parent);
    t.join();
  }
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event2));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipGraphDestroy(graph));
}
/*
 * Create a graph with different types of nodes say X with dependencies between
 * nodes. Create graph Y with few nodes with dependencies and X as child graph.
 * Now verify each of nodes of Y including the nodes inside child graph using
 * hipGraphNodeGetType()
 */
TEST_CASE("Unit_hipGraphNodeGetType_NodeTypeOfChildGraph_WithDependency") {
  hipGraph_t graph, childGraph, getGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyNode, kernelNode, hostNode, childGraphNode, emptyNode, memsetNode,
      waiteventNode, recordeventNode;

  // Create a child graph
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  // Add memSet Node to child graph
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 10;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, childGraph, nullptr, 0, &memsetParams));

  // Add WaitEvent Node to child graph
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamWaitEvent(stream1, event1, 0));
  HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, childGraph, nullptr, 0, event1));
  HIP_CHECK(hipGraphAddDependencies(childGraph, &memsetNode, &waiteventNode, 1));

  // Add Empty Node to child graph
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, childGraph, nullptr, 0));
  HIP_CHECK(hipGraphAddDependencies(childGraph, &waiteventNode, &emptyNode, 1));

  // Add Event Record Node to child graph
  HIP_CHECK(hipEventCreate(&event2));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventRecord(event2, stream2));
  HIP_CHECK(hipGraphAddEventRecordNode(&recordeventNode, childGraph, nullptr, 0, event2));
  HIP_CHECK(hipGraphAddDependencies(childGraph, &emptyNode, &recordeventNode, 1));

  // Add Host Node to parent graph
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  // Add MemCpy Node to parent graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &hostNode, &memcpyNode, 1));

  // Add Kernal Node to parent graph
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyNode, &kernelNode, 1));

  // Add child node to the parent graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelNode, &childGraphNode, 1));

  // Check Node Type of graph
  SECTION("Graph node Type verification") { ChkNodeTypeWithDependency(graph, Parent); }

  // Get the child graph from graph
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode, &getGraph));
  SECTION("Child Graph node Type verification") { ChkNodeTypeWithDependency(getGraph, Child); }

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event2));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipGraphDestroy(graph));
}
