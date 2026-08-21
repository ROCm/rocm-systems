/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Tests the device-memory footprint of a graph exec's internal stream pool.
 * Each hip::Stream a graph exec holds carries a VirtualGPU whose kernarg pool is
 * device-local on large-BAR parts, so a stream created per instantiate but never
 * used is charged against VRAM for the whole graph lifetime. Workloads that keep
 * thousands of graphs resident amplify this into gigabytes.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>

#include <vector>

namespace {

constexpr int kElems = 1024;

__global__ void bumpKernel(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += 1;
}

// Independent chains of kernel nodes, so the scheduler requests more than one
// internal stream rather than folding the graph onto a single one.
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
 *  - Instantiates many graphs and measures the device memory each one retains.
 *    A graph exec must not hold internal streams that its launches never use, so
 *    the steady-state cost per instantiate stays well below one stream's kernarg
 *    pool (HSA_KERNARG_POOL_SIZE, 4 MiB by default).
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphStreamPoolFootprint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphStreamPool_InstantiateFootprint) {
  constexpr int kGraphs = 64;
  constexpr size_t kMaxBytesPerGraph = 2 * 1024 * 1024;

  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, kElems * sizeof(int)));

  // Warm up so one-off runtime allocations are not attributed to the graphs.
  for (int i = 0; i < 4; ++i) {
    hipGraph_t warm_graph = nullptr;
    hipGraphExec_t warm_exec = nullptr;
    buildChainedGraph(&warm_graph, buf, 4, 2);
    HIP_CHECK(hipGraphInstantiate(&warm_exec, warm_graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphExecDestroy(warm_exec));
    HIP_CHECK(hipGraphDestroy(warm_graph));
  }

  const size_t free_before = freeDeviceMemory();

  std::vector<hipGraph_t> graphs(kGraphs, nullptr);
  std::vector<hipGraphExec_t> execs(kGraphs, nullptr);
  for (int i = 0; i < kGraphs; ++i) {
    buildChainedGraph(&graphs[i], buf, 4, 2);
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

  // Everything the graphs held must come back on destroy.
  const size_t free_end = freeDeviceMemory();
  const size_t retained = (free_before > free_end) ? (free_before - free_end) : 0;
  INFO("device memory retained after destroy: " << retained << " bytes");
  REQUIRE(retained <= kMaxBytesPerGraph);

  HIP_CHECK(hipFree(buf));
}

/**
 * Test Description
 * ------------------------
 *  - Interleaves same-device and cross-device launches of one graph exec. A
 *    cross-device launch needs an internal capture-device stream for slot 0,
 *    while a same-device launch supplies slot 0 itself; both orderings must keep
 *    working when that stream is created on demand rather than at instantiate.
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

  // same-device first, so slot 0 comes from the user stream
  HIP_CHECK(hipGraphLaunch(exec, same_dev_stream));
  HIP_CHECK(hipStreamSynchronize(same_dev_stream));

  // cross-device, which is where the internal slot-0 stream is needed
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipGraphLaunch(exec, cross_dev_stream));
  HIP_CHECK(hipStreamSynchronize(cross_dev_stream));
  HIP_CHECK(hipSetDevice(0));

  // back to same-device: the stream added above must be left idle
  HIP_CHECK(hipGraphLaunch(exec, same_dev_stream));
  HIP_CHECK(hipStreamSynchronize(same_dev_stream));

  // cross-device again, reusing rather than recreating
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipGraphLaunch(exec, cross_dev_stream));
  HIP_CHECK(hipStreamSynchronize(cross_dev_stream));
  HIP_CHECK(hipSetDevice(0));

  // Every launch must have run every node exactly once, on either device.
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
