/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Graph exec stream-pool device-memory footprint tests.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>

#include <cstdlib>
#include <vector>

namespace {

constexpr int kElems = 1024;

__global__ void bumpKernel(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += 1;
}

// Build chains of kernel nodes.
void buildChainedGraph(hipGraph_t* graph, int* buf, int chains, int depth) {
  HIP_CHECK(hipGraphCreate(graph, 0));
  for (int c = 0; c < chains; ++c) {
    hipGraphNode_t prev = nullptr;
    for (int d = 0; d < depth; ++d) {
      hipGraphNode_t node = nullptr;
      int elems = kElems;
      void* args[] = {&buf, &elems};
      hipKernelNodeParams params = {};
      params.func = reinterpret_cast<void*>(bumpKernel);
      params.gridDim = dim3(1);
      params.blockDim = dim3(64);
      params.sharedMemBytes = 0;
      params.kernelParams = args;
      params.extra = nullptr;
      HIP_CHECK(hipGraphAddKernelNode(&node, *graph, prev ? &prev : nullptr, prev ? 1 : 0,
                                      &params));
      prev = node;
    }
  }
}

size_t freeDeviceMemory() {
  size_t free_bytes = 0, total_bytes = 0;
  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
  return free_bytes;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Same-device instantiate footprint must stay below one kernarg pool. Single
 *    chain needs only the launch stream; no eager cross-device slot-0 stream.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphStreamPoolFootprint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphStreamPool_InstantiateFootprint) {
  constexpr int kGraphs = 64;
  constexpr int kChains = 1;
  constexpr int kDepth = 2;
  constexpr size_t kMaxBytesPerGraph = 2 * 1024 * 1024;

  const char* min_overlap = std::getenv("DEBUG_HIP_GRAPH_MIN_OVERLAP");
  INFO("DEBUG_HIP_GRAPH_MIN_OVERLAP=" << (min_overlap ? min_overlap : "(default)"));

  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, kElems * sizeof(int)));

  // Warm up one-off allocations away from the measurement.
  for (int i = 0; i < 4; ++i) {
    hipGraph_t warm_graph = nullptr;
    hipGraphExec_t warm_exec = nullptr;
    buildChainedGraph(&warm_graph, buf, kChains, kDepth);
    HIP_CHECK(hipGraphInstantiate(&warm_exec, warm_graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphExecDestroy(warm_exec));
    HIP_CHECK(hipGraphDestroy(warm_graph));
  }

  const size_t free_before = freeDeviceMemory();

  std::vector<hipGraph_t> graphs(kGraphs, nullptr);
  std::vector<hipGraphExec_t> execs(kGraphs, nullptr);
  for (int i = 0; i < kGraphs; ++i) {
    buildChainedGraph(&graphs[i], buf, kChains, kDepth);
    HIP_CHECK(hipGraphInstantiate(&execs[i], graphs[i], nullptr, nullptr, 0));
  }

  const size_t free_after = freeDeviceMemory();
  const size_t used = (free_before > free_after) ? (free_before - free_after) : 0;
  const size_t per_graph = used / kGraphs;
  INFO("device memory per instantiate: " << per_graph << " bytes over " << kGraphs << " graphs");
  REQUIRE(per_graph <= kMaxBytesPerGraph);

  for (int i = 0; i < kGraphs; ++i) {
    HIP_CHECK(hipGraphExecDestroy(execs[i]));
    HIP_CHECK(hipGraphDestroy(graphs[i]));
  }

  // Memory must return after destroy.
  const size_t free_end = freeDeviceMemory();
  const size_t retained = (free_before > free_end) ? (free_before - free_end) : 0;
  INFO("device memory retained after destroy: " << retained << " bytes");
  REQUIRE(retained <= kMaxBytesPerGraph);

  HIP_CHECK(hipFree(buf));
}

/**
 * Test Description
 * ------------------------
 *  - Same- and cross-device launches interleaved; slot-0 stream created on first
 *    cross-device launch, not at instantiate.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphStreamPoolFootprint.cc
 * Test requirements
 * ------------------------
 *  - Multi device
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphStreamPool_CrossDeviceInterleaved) {
  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);

  constexpr int kChains = 4;
  constexpr int kDepth = 2;
  constexpr int kLaunches = 4;

  HIP_CHECK(hipSetDevice(0));
  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, kElems * sizeof(int)));
  HIP_CHECK(hipMemset(buf, 0, kElems * sizeof(int)));

  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  buildChainedGraph(&graph, buf, kChains, kDepth);
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t same_dev_stream = nullptr;
  HIP_CHECK(hipStreamCreate(&same_dev_stream));
  hipStream_t cross_dev_stream = nullptr;
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipStreamCreate(&cross_dev_stream));
  HIP_CHECK(hipSetDevice(0));

  // same-device first (user stream is slot 0)
  HIP_CHECK(hipGraphLaunch(exec, same_dev_stream));
  HIP_CHECK(hipStreamSynchronize(same_dev_stream));

  // cross-device (creates internal slot-0 stream)
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipGraphLaunch(exec, cross_dev_stream));
  HIP_CHECK(hipStreamSynchronize(cross_dev_stream));
  HIP_CHECK(hipSetDevice(0));

  HIP_CHECK(hipGraphLaunch(exec, same_dev_stream));
  HIP_CHECK(hipStreamSynchronize(same_dev_stream));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipGraphLaunch(exec, cross_dev_stream));
  HIP_CHECK(hipStreamSynchronize(cross_dev_stream));
  HIP_CHECK(hipSetDevice(0));

  int observed = 0;
  HIP_CHECK(hipMemcpy(&observed, buf, sizeof(int), hipMemcpyDeviceToHost));
  INFO("observed " << observed << " increments");
  REQUIRE(observed == kChains * kDepth * kLaunches);

  HIP_CHECK(hipStreamDestroy(same_dev_stream));
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipStreamDestroy(cross_dev_stream));
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(buf));
}
