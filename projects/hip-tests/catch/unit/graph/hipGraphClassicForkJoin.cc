/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Classic-path graph that catches a skipped stream id.
 *
 * Nodes are added A, B, D, C, E, F. C's outgoing edges are E, then D, then F.
 * D is already reachable through B. A scheduler that advances the stream id
 * after every edge, including one whose target is already scheduled, assigns
 * F id 3 and leaves id 2 unused. The pool is sized by the count of ids in use
 * ({0, 1, 3} is 3), so SetStream reads streams_[3] and aborts.
 *
 * ScheduleOneNode assigns contiguous ids, so this launch succeeds. The same
 * graph fails if a later scheduler skips an id.
 *
 * DEBUG_HIP_GRAPH_CLASSIC_PATH is read when the runtime loads.
 */

#include <hip_test_common.hh>

#include <cstdlib>
#include <string>

static __global__ void writeVal(int* p, int v) { *p = v; }

HIP_TEST_CASE(Unit_hipGraphClassic_ForkJoin_StreamPool) {
  const char* classic = std::getenv("DEBUG_HIP_GRAPH_CLASSIC_PATH");
  REQUIRE(classic != nullptr);
  REQUIRE(std::string(classic) == "1");

  // Slots: A B D C E F
  constexpr int kNodes = 6;
  constexpr int kValues[kNodes] = {1, 2, 3, 4, 5, 6};

  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, kNodes * sizeof(int)));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  auto addWrite = [&](hipGraphNode_t* node, int slot, hipGraphNode_t* deps, size_t depCount) {
    int* ptr = buf + slot;
    int val = kValues[slot];
    void* args[] = {&ptr, &val};
    hipKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(writeVal);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(node, graph, deps, depCount, &params));
  };

  hipGraphNode_t nodeA = nullptr;
  hipGraphNode_t nodeB = nullptr;
  hipGraphNode_t nodeD = nullptr;
  hipGraphNode_t nodeC = nullptr;
  hipGraphNode_t nodeE = nullptr;
  hipGraphNode_t nodeF = nullptr;

  addWrite(&nodeA, 0, nullptr, 0);
  addWrite(&nodeB, 1, &nodeA, 1);
  addWrite(&nodeD, 2, &nodeB, 1);
  addWrite(&nodeC, 3, &nodeA, 1);
  addWrite(&nodeE, 4, &nodeC, 1);
  // C's edges must be E, then D, then F. D is already on the graph.
  HIP_CHECK(hipGraphAddDependencies(graph, &nodeC, &nodeD, 1));
  addWrite(&nodeF, 5, &nodeC, 1);

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipMemsetAsync(buf, 0, kNodes * sizeof(int), stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int host[kNodes] = {};
  HIP_CHECK(hipMemcpy(host, buf, sizeof(host), hipMemcpyDeviceToHost));
  for (int i = 0; i < kNodes; ++i) {
    REQUIRE(host[i] == kValues[i]);
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(buf));
}
