/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
/* Test verifies hipGraphLaunch API
Negative scenarios -
1) Pass graphExec as nullptr and verify api returns error code.
2) Pass pGraphExec as nullptr and stream as hipStreamPerThread and verify  api returns error code.
3) Pass pGraphExec as empty object and verify  api returns error code.
4) Destroy executable graph and try to launch it. Make sure api should not crash and it should
returns error code. 5) Destroy stream and try to launch respective executable graph. Make sure api
should not crash and it should returns error code. 6) Destroy actual graph created and try to launch
respective executable graph. Check api should execute properly without crash or error code.
Functional Scenario -
1) Check basic functionality with stream as hipStreamPerThread
2) Test hipGraphLaunch call on multiple devices.
3) Create a graph with multiple nodes. Create an executable graph.
   Launch the executable graph 3 times in stream simultaneously.
   Wait for stream. Validate the output. No issues should be observed
4) Create a graph with multiple nodes. Create an executable graph.
   Verify if an executable graph be launched on null stream.
*/

#define SIZE 1024
#define TEST_LOOP_SIZE 3

static void hipGraphLaunch_test() {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  constexpr size_t updateVal = 1;
  char *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  char *A_h{nullptr}, *B_h{nullptr};

  HipTest::initArrays<char>(&A_d, &B_d, &C_d, &A_h, &B_h, nullptr, N, false);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t memsetNode;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(C_d);
  memsetParams.value = val;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));

  std::vector<hipGraphNode_t> dependencies;
  dependencies.push_back(memsetNode);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = updateVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, dependencies.data(), dependencies.size(),
                                  &memsetParams));
  HIP_CHECK(hipGraphMemsetNodeSetParams(memsetNode, &memsetParams));
  dependencies.push_back(memsetNode);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

  // Validating the result
  for (size_t i = 0; i < Nbytes; i++) {
    if (A_h[i] != updateVal) {
      WARN("Validation failed at- " << i << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<char>(A_d, B_d, C_d, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

HIP_TEST_CASE(Unit_hipGraphLaunch_Functional_multidevice_test) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));
      hipGraphLaunch_test();
    }
  } else {
    SUCCEED("Skipped the testcase as there is no device to test.");
  }
}

// Function to fill input data
static void fillRandInpData(int* A1_h, int* A2_h, size_t N) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < N; i++) {
    A1_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
    A2_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}
// Function to validate result
static void validateOutData(int* A1_h, int* A2_h, size_t N) {
  for (size_t i = 0; i < N; i++) {
    int result = (A1_h[i] * A1_h[i]);
    REQUIRE(result == A2_h[i]);
  }
}
/*
 * 1.Create a graph with multiple nodes. Create an executable graph.
 * Launch the executable graph 3 times in stream simultaneously.
 * Wait for stream. Validate the output. No issues should be observed
 * 2.Create a graph with multiple nodes. Create an executable graph.
 * Verify if an executable graph be launched on null stream.
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_Functional_MultipleLaunch) {
  size_t memSize = SIZE;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, SIZE);
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  int *A_h{nullptr}, *A_d{nullptr}, *C_d{nullptr}, *C_h{nullptr};

  HipTest::initArrays<int>(&A_d, &C_d, nullptr, &A_h, &C_h, nullptr, SIZE, false);

  hipGraphNode_t memcpyH2D, memcpyD2H, kernelNode;

  // Create memcpy H2D nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph, nullptr, 0, A_d, A_h, (sizeof(int) * SIZE),
                                    hipMemcpyHostToDevice));
  nodeDependencies.push_back(memcpyH2D);
  // Creating kernel node
  hipKernelNodeParams kerNodeParams;
  void* kernelArgs[] = {reinterpret_cast<void*>(&A_d), reinterpret_cast<void*>(&C_d),
                        reinterpret_cast<void*>(&memSize)};
  kerNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams.gridDim = dim3(blocks);
  kerNodeParams.blockDim = dim3(threadsPerBlock);
  kerNodeParams.sharedMemBytes = 0;
  kerNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kerNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &kerNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  // Create memcpy D2H nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, nodeDependencies.data(),
                                    nodeDependencies.size(), C_h, C_d, (sizeof(int) * SIZE),
                                    hipMemcpyDeviceToHost));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  // Execute graph
  SECTION("Multiple Graph Launch") {
    for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
      fillRandInpData(A_h, C_h, SIZE);
      HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
      HIP_CHECK(hipStreamSynchronize(streamForGraph));
      validateOutData(A_h, C_h, SIZE);
    }
  }
  SECTION("Graph launch on Null stream") {
    for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
      fillRandInpData(A_h, C_h, SIZE);
      HIP_CHECK(hipGraphLaunch(graphExec, 0));
      HIP_CHECK(hipStreamSynchronize(0));
      validateOutData(A_h, C_h, SIZE);
    }
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));

  // Free
  HipTest::freeArrays<int>(A_d, C_d, nullptr, A_h, C_h, nullptr, false);
}
