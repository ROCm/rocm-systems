/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Gated fan-out that drops a cross-stream wait when out-edge index i >= Q.
 *
 * Root feeds C0..C11 in that edge order. With DEBUG_HIP_FORCE_GRAPH_QUEUES=4,
 * child i is placed on stream i % 4. C1, C2, and C3 also depend on Gate, so a
 * queue walk can submit C5 (stream 1, wait flag clear) before C1 installs the
 * barrier on that stream. Each child records whether its parents had already
 * published when the child started. A missing wait leaves that value 0.
 *
 * The same graph is launched on the classic path and on the segmented path.
 * DEBUG_HIP_GRAPH_CLASSIC_PATH is read when the runtime loads, so each path is
 * a separate process.
 */

#include <hip_test_common.hh>

#include <cstdlib>
#include <string>

static constexpr int kFanout = 12;
static constexpr int kLaunches = 5;
static constexpr unsigned long long kSpins = 5000000ull;

static __global__ void publish(int* flag, unsigned long long spins) {
  unsigned long long start = clock64();
  while (clock64() - start < spins) {
  }
  atomicExch(flag, 1);
}

static __global__ void observe(int* first, int* second, int useSecond, int* saw,
                               unsigned long long spins) {
  int ready = atomicAdd(first, 0);
  if (useSecond) {
    ready &= atomicAdd(second, 0);
  }
  atomicExch(saw, ready);
  unsigned long long start = clock64();
  while (clock64() - start < spins) {
  }
}

static void RunGatedFanout() {
  const char* queues = std::getenv("DEBUG_HIP_FORCE_GRAPH_QUEUES");
  REQUIRE(queues != nullptr);
  REQUIRE(std::string(queues) == "4");

  // slots: root, gate, then one observation per child
  constexpr int kSlots = 2 + kFanout;
  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, kSlots * sizeof(int)));
  int* rootFlag = buf;
  int* gateFlag = buf + 1;
  int* saw = buf + 2;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  auto addPublish = [&](hipGraphNode_t* node, int* flag) {
    unsigned long long spins = kSpins;
    void* args[] = {&flag, &spins};
    hipKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(publish);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(node, graph, nullptr, 0, &params));
  };

  hipGraphNode_t root = nullptr;
  hipGraphNode_t gate = nullptr;
  addPublish(&root, rootFlag);
  addPublish(&gate, gateFlag);

  hipGraphNode_t child[kFanout] = {};
  for (int i = 0; i < kFanout; ++i) {
    const bool gated = (i == 1 || i == 2 || i == 3);
    hipGraphNode_t deps[2] = {root, gate};
    int* childSaw = saw + i;
    unsigned long long spins = kSpins;
    int useSecond = gated ? 1 : 0;
    int* second = gated ? gateFlag : rootFlag;
    void* args[] = {&rootFlag, &second, &useSecond, &childSaw, &spins};
    hipKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(observe);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&child[i], graph, deps, gated ? 2 : 1, &params));
  }

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int launch = 0; launch < kLaunches; ++launch) {
    HIP_CHECK(hipMemsetAsync(buf, 0, kSlots * sizeof(int), stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int host[kFanout] = {};
    HIP_CHECK(hipMemcpy(host, saw, sizeof(host), hipMemcpyDeviceToHost));
    for (int i = 0; i < kFanout; ++i) {
      INFO("launch " << launch << " child " << i);
      REQUIRE(host[i] == 1);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(buf));
}

HIP_TEST_CASE(Unit_hipGraphGatedFanout_Order_Classic) {
  const char* classic = std::getenv("DEBUG_HIP_GRAPH_CLASSIC_PATH");
  REQUIRE(classic != nullptr);
  REQUIRE(std::string(classic) == "1");
  RunGatedFanout();
}

HIP_TEST_CASE(Unit_hipGraphGatedFanout_Order_Segmented) {
  const char* classic = std::getenv("DEBUG_HIP_GRAPH_CLASSIC_PATH");
  REQUIRE(classic == nullptr);
  RunGatedFanout();
}
