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

/**
Testcase Scenarios :
Functional -
1) Create a graph and then used it for hipGraphInstantiate without adding any node to graph.
2.a) Create an invalid graph as shown below:
   ---> empty node ---> empty node --->
     ^                              |
     |                              |
     --------------------------------
   Try instantiating the graph. Instantiating the graph should result in error.
2.b) Create a more complex cyclic graph. Try instantiating the graph. Instantiating the graph
   should result in error.
2.c) Create a graph with child node. The graph in the child node is a cyclical graph.
   Try instantiating the graph. Instantiating the graph should result in error.
3.a) Create a graph with redundant dependencies. Instantiate and execute the graph and validate
   the output.
3.b) Create a graph. Instantiate the graph multiple times. Execute all the instantiated graphs
   and validate the output. Destroy the instantiated graphs at the end.
3.c) Create a graph. Instantiate the graph multiple times. In loop, execute an instantiated
   graph, validate the output and destroy the current instantiated graph.

Negative -
1) Pass pGraphExec as null ptr and verify that api returns error code and doesn’t crash.
2) Pass graph as null/invalid ptr and check if api returns error.
3) Pass pGraphExec as un-initilize object and verify that api returns error code and doesn’t crash.
4) Pass Graph as un-initilize and verify that api returns error code and doesn’t crash.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define NUM_OF_INSTANCES 10
/* Test verifies hipGraphInstantiate API Negative scenarios.
 */

TEST_CASE("Unit_hipGraphInstantiate_Negative") {
  hipError_t ret;
  hipGraphExec_t gExec{};
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  SECTION("Pass pGraphExec as nullptr") {
    ret = hipGraphInstantiate(nullptr, graph, nullptr, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass graph as null/invalid ptr") {
    ret = hipGraphInstantiate(&gExec, nullptr, nullptr, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Graph as un-initialize") {
    hipGraph_t graph_uninit{};
    ret = hipGraphInstantiate(&gExec, graph_uninit, nullptr, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pGraphExec as un-initialize") {
    ret = hipGraphInstantiate(&gExec, graph, nullptr, nullptr, 0);
    REQUIRE(hipSuccess == ret);
  }
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipGraphInstantiate Basic scenarios.
Create a graph and then used it for hipGraphInstantiate without adding any node to graph.
 */
TEST_CASE("Unit_hipGraphInstantiate_Basic") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  REQUIRE(nullptr != graph);
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}
#if HT_NVIDIA
/* Test Functional Scenario 2.a, 2.b, 2.c with hipGraphInstantiate and
hipGraphInstantiateWithFlags.
*/
TEST_CASE("Unit_hipGraphInstantiate_InvalidCyclicGraph") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  SECTION("Simple Cyclic Graph") {
    hipGraphNode_t emptyNode1, emptyNode2;
    hipGraph_t clonedgraph;
    // Create emptyNode and add it to graph with dependency
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nullptr, 0));
    // Create illegal dependency
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &emptyNode2, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode2, &emptyNode1, 1));
    // Detect the error during instantiation
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiateWithFlags(&graphExec, graph, 0));
    // Clone the illegal graph
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    // Try instantiating the cloned graph
    REQUIRE(hipErrorInvalidValue ==
            hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiateWithFlags(&graphExec, clonedgraph, 0));
  }

  SECTION("A More Complex Cyclic Graph") {
    hipGraphNode_t emptyNode1, emptyNode2, emptyNode3, emptyNode4, emptyNode5, emptyNode6,
        emptyNode7;
    hipGraph_t clonedgraph;
    // Create emptyNode and add it to graph with dependency
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode3, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode4, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode5, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode6, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode7, graph, nullptr, 0));
    // Create illegal dependency
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &emptyNode2, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode2, &emptyNode3, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode3, &emptyNode4, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode4, &emptyNode7, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &emptyNode5, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode5, &emptyNode6, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode6, &emptyNode7, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode4, &emptyNode1, 1));
    // Detect the error during instantiation
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiateWithFlags(&graphExec, graph, 0));
    // Clone the illegal graph
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    // Try instantiating the cloned graph
    REQUIRE(hipErrorInvalidValue ==
            hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiateWithFlags(&graphExec, clonedgraph, 0));
  }

  SECTION("A Cyclic Graph as Child Node") {
    hipGraph_t childgraph;
    HIP_CHECK(hipGraphCreate(&childgraph, 0));
    hipGraphNode_t emptyNode1, emptyNode2, emptyNode3, emptyNode4, emptyNode5, emptyNode6,
        childNode;
    // Create emptyNode and add it to graph with dependency
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nullptr, 0));
    // Create emptyNode and add it to childgraph with dependency
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode3, childgraph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode4, childgraph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode5, childgraph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode6, childgraph, nullptr, 0));
    HIP_CHECK(hipGraphAddDependencies(childgraph, &emptyNode3, &emptyNode4, 1));
    HIP_CHECK(hipGraphAddDependencies(childgraph, &emptyNode4, &emptyNode5, 1));
    HIP_CHECK(hipGraphAddDependencies(childgraph, &emptyNode5, &emptyNode6, 1));
    // Illegal dependency
    HIP_CHECK(hipGraphAddDependencies(childgraph, &emptyNode5, &emptyNode4, 1));
    HIP_CHECK(hipGraphAddChildGraphNode(&childNode, graph, nullptr, 0, childgraph));
    HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &childNode, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &childNode, &emptyNode2, 1));
    // Detect the error during instantiation
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    REQUIRE(hipErrorInvalidValue == hipGraphInstantiateWithFlags(&graphExec, graph, 0));
  }
  HIP_CHECK(hipGraphDestroy(graph));
}
#endif
/* Local function to initialize input data.
 */
static void init_input(int* a, size_t size) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < size; i++) {
    a[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}

/* Test Functional Scenario 3.a, 3.b and 3.c.
 */
TEST_CASE("Unit_hipGraphInstantiate_functionalScenarios") {
  hipGraph_t graph;
  hipGraphExec_t graphExec[NUM_OF_INSTANCES];
  HIP_CHECK(hipGraphCreate(&graph, 0));

  constexpr size_t size = 1024;
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, size);
  hipGraphNode_t memcpyh2d, kernelNode, memcpyd2h;
  int *inputVec_d{nullptr}, *inputVec_h{nullptr}, *outputVec_h{nullptr}, *outputVec_d{nullptr};
  // host and device allocation
  HipTest::initArrays<int>(&inputVec_d, &outputVec_d, nullptr, &inputVec_h, &outputVec_h, nullptr,
                           size, false);
  // Create graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyh2d, graph, nullptr, 0, inputVec_d, inputVec_h,
                                    sizeof(int) * size, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyd2h, graph, nullptr, 0, outputVec_h, outputVec_d,
                                    sizeof(int) * size, hipMemcpyDeviceToHost));

  hipKernelNodeParams kernelNodeParams{};
  size_t N = size;
  void* kernelArgs[3] = {reinterpret_cast<void*>(&inputVec_d),
                         reinterpret_cast<void*>(&outputVec_d), reinterpret_cast<void*>(&N)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kernelNodeParams.gridDim = dim3(blocks, 1, 1);
  kernelNodeParams.blockDim = dim3(threadsPerBlock, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyh2d, &kernelNode, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelNode, &memcpyd2h, 1));

  SECTION("Creating Redundant Dependencies") {
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyh2d, &memcpyd2h, 1));
    // Create Executable Graphs
    HIP_CHECK(hipGraphInstantiate(&graphExec[0], graph, nullptr, nullptr, 0));
    REQUIRE(graphExec[0] != nullptr);
    // Test Graph
    init_input(inputVec_h, size);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipGraphLaunch(graphExec[0], stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t i = 0; i < size; i++) {
      REQUIRE(outputVec_h[i] == (inputVec_h[i] * inputVec_h[i]));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipGraphExecDestroy(graphExec[0]));
  }

  SECTION("Creating Multiple Instances Graph") {
    // Create Executable Graphs
    for (int i = 0; i < NUM_OF_INSTANCES; i++) {
      HIP_CHECK(hipGraphInstantiate(&graphExec[i], graph, nullptr, nullptr, 0));
      REQUIRE(graphExec[i] != nullptr);
    }
    // Execute all the instances of the graph
    init_input(inputVec_h, size);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    for (int i = 0; i < NUM_OF_INSTANCES; i++) {
      HIP_CHECK(hipGraphLaunch(graphExec[i], stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      for (size_t ii = 0; ii < size; ii++) {
        REQUIRE(outputVec_h[ii] == (inputVec_h[ii] * inputVec_h[ii]));
      }
    }
    HIP_CHECK(hipStreamDestroy(stream));
    for (int i = 0; i < NUM_OF_INSTANCES; i++) {
      HIP_CHECK(hipGraphExecDestroy(graphExec[i]));
    }
  }

  SECTION("Creating Multiple Instances Graph and Destroying After Use") {
    // Create Executable Graphs
    for (int i = 0; i < NUM_OF_INSTANCES; i++) {
      HIP_CHECK(hipGraphInstantiate(&graphExec[i], graph, nullptr, nullptr, 0));
      REQUIRE(graphExec[i] != nullptr);
    }
    // Execute all the instances of the graph
    init_input(inputVec_h, size);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    for (int i = 0; i < NUM_OF_INSTANCES; i++) {
      HIP_CHECK(hipGraphLaunch(graphExec[i], stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      for (size_t ii = 0; ii < size; ii++) {
        REQUIRE(outputVec_h[ii] == (inputVec_h[ii] * inputVec_h[ii]));
      }
      HIP_CHECK(hipGraphExecDestroy(graphExec[i]));
    }
    HIP_CHECK(hipStreamDestroy(stream));
  }
  // Free
  HipTest::freeArrays<int>(inputVec_d, outputVec_d, nullptr, inputVec_h, outputVec_h, nullptr,
                           false);
  HIP_CHECK(hipGraphDestroy(graph));
}
