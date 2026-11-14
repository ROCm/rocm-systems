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
2) In a chain of kernel nodes with the middle node disabled, update the first and the
   last node and check the updates take effect both while the middle node is disabled
   and after it is re-enabled.
3) Disable the head of a cross-stream batch so its fallback dependency barrier occupies
   the first filtered slot, update a later enabled kernel in that batch, and check the
   new params and the dependency wait.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include <chrono>
#include <thread>

__global__ void ExecKernelNodeSetParamsStore(int* output, int value) { output[0] = value; }

__global__ void ExecKernelNodeSetParamsDelayStore(int* output, int value, unsigned long long ticks) {
#if HT_AMD
  const unsigned long long start = wall_clock64();
  while (wall_clock64() - start < ticks) {
  }
#else
  const unsigned long long start = clock64();
  while (clock64() - start < ticks) {
  }
#endif
  output[0] = value;
}

// Copies |src| and then stores |value|, so a lost dependency wait observes the cleared source.
__global__ void ExecKernelNodeSetParamsCopyStore(const int* src, int* dst, int value) {
  dst[0] = src[0];
  dst[1] = value;
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

constexpr int kChainNodes = 3;

// Launches |exec| a few times and checks the value each chain node stored in |output|.
static void LaunchChainAndCheck(hipGraphExec_t exec, hipStream_t stream, int* output,
                                const int (&expected)[kChainNodes]) {
  constexpr int kLaunches = 3;
  for (int launch = 0; launch < kLaunches; ++launch) {
    HIP_CHECK(hipMemset(output, 0, kChainNodes * sizeof(int)));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    int actual[kChainNodes] = {};
    HIP_CHECK(hipMemcpy(actual, output, sizeof(actual), hipMemcpyDeviceToHost));
    for (int i = 0; i < kChainNodes; ++i) {
      INFO("node " << i);
      REQUIRE(actual[i] == expected[i]);
    }
  }
}

/**
 * Functional Test: update the nodes of a single-stream kernel chain while a node between them
 * is disabled. The chain keeps all three nodes in one packet batch, so the updates rewrite the
 * batch's dispatch buffer that skips the disabled node, including its first slot.
 */
HIP_TEST_CASE(Unit_hipGraphExecKernelNodeSetParams_DisabledNeighborInBatch) {
  constexpr int kNodes = kChainNodes;
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kNodes * sizeof(int));
  int* ptrs[kNodes];
  int values[kNodes] = {1, 2, 3};
  void* args[kNodes][2];
  hipKernelNodeParams params[kNodes];
  hipGraphNode_t nodes[kNodes];

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  for (int i = 0; i < kNodes; ++i) {
    ptrs[i] = output.ptr() + i;
    args[i][0] = &ptrs[i];
    args[i][1] = &values[i];
    params[i] = hipKernelNodeParams{};
    params[i].func = reinterpret_cast<void*>(ExecKernelNodeSetParamsStore);
    params[i].gridDim = dim3(1);
    params[i].blockDim = dim3(1);
    params[i].kernelParams = args[i];
    HIP_CHECK(hipGraphAddKernelNode(&nodes[i], graph, i == 0 ? nullptr : &nodes[i - 1],
                                    i == 0 ? 0 : 1, &params[i]));
  }

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  LaunchChainAndCheck(exec, stream, output.ptr(), {1, 2, 3});

  HIP_CHECK(hipGraphNodeSetEnabled(exec, nodes[1], 0));
  LaunchChainAndCheck(exec, stream, output.ptr(), {1, 0, 3});

  values[2] = 30;
  HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, nodes[2], &params[2]));
  LaunchChainAndCheck(exec, stream, output.ptr(), {1, 0, 30});

  values[0] = 10;
  HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, nodes[0], &params[0]));
  LaunchChainAndCheck(exec, stream, output.ptr(), {10, 0, 30});

  HIP_CHECK(hipGraphNodeSetEnabled(exec, nodes[1], 1));
  LaunchChainAndCheck(exec, stream, output.ptr(), {10, 2, 30});

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

// Fails the test when a lost completion signal would otherwise hang the suite.
static void LaunchAndWaitBounded(hipGraphExec_t exec, hipStream_t stream) {
  HIP_CHECK(hipGraphLaunch(exec, stream));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  hipError_t query = hipErrorNotReady;
  while ((query = hipStreamQuery(stream)) == hipErrorNotReady &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  REQUIRE(query == hipSuccess);
}

/**
 * The batch under test is three kernels that wait on one spinning kernel on another stream.
 * Disabling its first node splices the reserved dependency barrier into filtered slot 0.
 * Disabling the middle node as well puts the last kernel's filtered slot off its dispatch
 * index, so the in-place param update has to use that slot. The last kernel copies the
 * producer's output: a rewrite that lands on the barrier leaves the old value or lets the
 * kernel run before the producer. Forks ahead of the batch make the segment graph deeper
 * than four levels, so the default scheduler does not collapse it onto one stream.
 */
HIP_TEST_CASE(Unit_hipGraphExecKernelNodeSetParams_DisabledBatchHead) {
  constexpr int kSteps = 6;
  constexpr int kProducer = 40;
  constexpr int kHead = 1;
  constexpr int kMiddle = 2;
  constexpr int kLast = 3;
  constexpr int kLastUpdated = 30;
  constexpr unsigned long long kSpinUs = 2000;
  // [0] producer, [1] copy of producer, [2] last kernel's value, [3] head, [4] middle, [5] scratch
  constexpr int kOutputs = 6;
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kOutputs * sizeof(int));
  int* producer_ptr = output.ptr();
  int* copy_ptr = output.ptr() + 1;
  int* head_ptr = output.ptr() + 3;
  int* middle_ptr = output.ptr() + 4;
  int* scratch_ptr = output.ptr() + 5;
  int scratch = 0;
  int producer = kProducer;
  int head_value = kHead;
  int middle_value = kMiddle;
  int last_value = kLast;
  unsigned long long ticks = kSpinUs * SpinTicksPerUs();
  void* scratch_args[] = {&scratch_ptr, &scratch};
  void* producer_args[] = {&producer_ptr, &producer, &ticks};
  void* head_args[] = {&head_ptr, &head_value};
  void* middle_args[] = {&middle_ptr, &middle_value};
  void* last_args[] = {&producer_ptr, &copy_ptr, &last_value};

  hipKernelNodeParams store{};
  store.func = reinterpret_cast<void*>(ExecKernelNodeSetParamsStore);
  store.gridDim = dim3(1);
  store.blockDim = dim3(1);
  store.kernelParams = scratch_args;
  hipKernelNodeParams delay = store;
  delay.func = reinterpret_cast<void*>(ExecKernelNodeSetParamsDelayStore);
  delay.kernelParams = producer_args;
  hipKernelNodeParams head_params = store;
  head_params.kernelParams = head_args;
  hipKernelNodeParams middle_params = store;
  middle_params.kernelParams = middle_args;
  hipKernelNodeParams last_params{};
  last_params.func = reinterpret_cast<void*>(ExecKernelNodeSetParamsCopyStore);
  last_params.gridDim = dim3(1);
  last_params.blockDim = dim3(1);
  last_params.kernelParams = last_args;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t chain[kSteps];
  hipGraphNode_t side[kSteps];
  // Continuation is added before the side branch, so the chain segment is first at its level
  // and stays on stream 0. The last chain node then forks to a filler first and the batch
  // second, which puts the batch on the other stream.
  HIP_CHECK(hipGraphAddKernelNode(&chain[0], graph, nullptr, 0, &store));
  for (int i = 0; i < kSteps - 2; ++i) {
    HIP_CHECK(hipGraphAddKernelNode(&chain[i + 1], graph, &chain[i], 1, &store));
    HIP_CHECK(hipGraphAddKernelNode(&side[i], graph, &chain[i], 1, &store));
  }
  HIP_CHECK(hipGraphAddKernelNode(&chain[kSteps - 1], graph, &chain[kSteps - 2], 1, &delay));
  HIP_CHECK(hipGraphAddKernelNode(&side[kSteps - 2], graph, &chain[kSteps - 2], 1, &store));
  HIP_CHECK(hipGraphAddKernelNode(&side[kSteps - 1], graph, &chain[kSteps - 1], 1, &store));

  hipGraphNode_t head = nullptr;
  hipGraphNode_t middle = nullptr;
  hipGraphNode_t last = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&head, graph, &chain[kSteps - 1], 1, &head_params));
  HIP_CHECK(hipGraphAddKernelNode(&middle, graph, &head, 1, &middle_params));
  HIP_CHECK(hipGraphAddKernelNode(&last, graph, &middle, 1, &last_params));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  auto check = [&](int copied, int last_stored, int head_stored, int middle_stored) {
    for (int launch = 0; launch < 3; ++launch) {
      HIP_CHECK(hipMemset(output.ptr(), 0, kOutputs * sizeof(int)));
      LaunchAndWaitBounded(exec, stream);
      int actual[kOutputs] = {};
      HIP_CHECK(hipMemcpy(actual, output.ptr(), sizeof(actual), hipMemcpyDeviceToHost));
      INFO("launch " << launch);
      REQUIRE(actual[0] == kProducer);
      REQUIRE(actual[1] == copied);
      REQUIRE(actual[2] == last_stored);
      REQUIRE(actual[3] == head_stored);
      REQUIRE(actual[4] == middle_stored);
    }
  };

  check(kProducer, kLast, kHead, kMiddle);

  HIP_CHECK(hipGraphNodeSetEnabled(exec, head, 0));
  HIP_CHECK(hipGraphNodeSetEnabled(exec, middle, 0));
  check(kProducer, kLast, 0, 0);

  last_value = kLastUpdated;
  HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, last, &last_params));
  check(kProducer, kLastUpdated, 0, 0);

  HIP_CHECK(hipGraphNodeSetEnabled(exec, head, 1));
  HIP_CHECK(hipGraphNodeSetEnabled(exec, middle, 1));
  check(kProducer, kLastUpdated, kHead, kMiddle);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}
