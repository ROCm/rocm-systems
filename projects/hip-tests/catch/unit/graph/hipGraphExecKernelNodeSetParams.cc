/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Test Case Scenarios :
Negative -
1) Pass hGraphExec as nullptr and verify api returns error code.
2) Pass node as nullptr and verify api returns error code.
3) Pass NodeParams as un-initialized structure object and verify api returns error code.
4) Pass pNodeParams as nullptr and verify api returns error code.
5) Pass NodeParams:func data member as nullptr and verify api returns error code.
Functional -
1) Instantiate a graph with kernel node, obtain executable graph and update
   the kernel node params with set and check it is taking effect.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

namespace {
struct GraphUpdateArguments {
  int values[256];
};

__global__ void graphWriteSmall(int* out, int value) { *out = value; }

__global__ void graphWriteLarge(int* out, GraphUpdateArguments args) {
  int sum = 0;
  for (int i = 0; i < 256; ++i) {
    sum += args.values[i];
  }
  *out = sum;
}
}  // namespace

// A replacement kernel can need more kernarg storage than the entire initial
// graph. Existing nodes must keep their arguments when the pool grows.
HIP_TEST_CASE(Unit_hipGraphExecKernelNodeSetParams_LargerArguments) {
  hipGraph_t graph;
  hipGraphExec_t exec;
  hipGraphNode_t first, second;
  hipStream_t stream;
  int* output;
  HIP_CHECK(hipMalloc(&output, 2 * sizeof(int)));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int value = 7;
  int sentinel = 1234;
  int* second_output = output + 1;
  void* small_args[] = {&output, &value};
  void* second_args[] = {&second_output, &sentinel};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(graphWriteSmall);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.kernelParams = small_args;
  HIP_CHECK(hipGraphAddKernelNode(&first, graph, nullptr, 0, &params));
  params.kernelParams = second_args;
  HIP_CHECK(hipGraphAddKernelNode(&second, graph, &first, 1, &params));
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  GraphUpdateArguments large{};
  void* large_args[] = {&output, &large};
  for (int iteration = 0; iteration < 64; ++iteration) {
    value = iteration + 1;
    for (int& element : large.values) {
      element = value;
    }
    const bool use_large = (iteration % 2 == 0);
    params.func = use_large ? reinterpret_cast<void*>(graphWriteLarge)
                            : reinterpret_cast<void*>(graphWriteSmall);
    params.kernelParams = use_large ? large_args : small_args;
    HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, first, &params));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    int result[2];
    HIP_CHECK(hipMemcpy(result, output, sizeof(result), hipMemcpyDeviceToHost));
    REQUIRE(result[0] == value * (use_large ? 256 : 1));
    REQUIRE(result[1] == sentinel);
  }
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(output));
}

/**
 * Negative Test for API hipGraphExecKernelNodeSetParams
 */
HIP_TEST_CASE(Unit_hipGraphExecKernelNodeSetParams_Negative) {
  constexpr size_t N = 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraph_t graph;
  hipGraphNode_t kNode{};
  hipKernelNodeParams kNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));

  hipGraphNode_t empty_node;
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, &kNode, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Pass hipGraphExec as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(nullptr, kNode, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass Node as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, nullptr, &kNodeParams),
                    hipErrorInvalidValue);
  }

#if HT_AMD
  /* NodeParams null check is disabled on Nvidia as
   * this call gives SIGSEGV error in CUDA setup */
  SECTION("Pass NodeParams as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, nullptr),
                    hipErrorInvalidValue);
  }
#endif

#if HT_NVIDIA  // on AMD this returns hipErrorInvalidValue
  SECTION("Pass NodeParams func data member as nullptr") {
    kNodeParams.func = nullptr;
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams),
                    hipErrorInvalidDeviceFunction);
  }
#endif

#if HT_NVIDIA  // segfaults on AMD
  SECTION("Pass kernelParams data member as nullptr") {
    kNodeParams.kernelParams = nullptr;
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams),
                    hipErrorInvalidValue);
  }
#endif

#if HT_NVIDIA  // segfaults on AMD
  SECTION("node is not a kernel node") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, empty_node, &kNodeParams),
                    hipErrorInvalidValue);
  }
#endif

  SECTION("node is not instantiated") {
    HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams),
                    hipErrorInvalidValue);
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Functional Test for API Exec Kernel Params
 */
HIP_TEST_CASE(Unit_hipGraphExecKernelNodeSetParams_Functional) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode;
  hipKernelNodeParams kNodeParams{}, kNodeParams1{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  HIP_CHECK(
      hipGraphAddKernelNode(&kNode, graph, dependencies.data(), dependencies.size(), &kNodeParams));

  kNodeParams1.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kNodeParams1.gridDim = dim3(blocks);
  kNodeParams1.blockDim = dim3(threadsPerBlock);
  kNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs);

  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(), dependencies.size(),
                                    C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphNodeSetEnabled(graphExec, kNode, 0));
  HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams1));
  HIP_CHECK(hipMemset(C_d, 0, Nbytes));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipGraphNodeSetEnabled(graphExec, kNode, 1));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorSUB<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
