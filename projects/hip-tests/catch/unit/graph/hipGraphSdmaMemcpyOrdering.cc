/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdlib>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 */

namespace {
constexpr int kBranches = 4;
constexpr size_t kElems = 2u << 20;  // 8 MiB per buffer
constexpr int kBlock = 256;
constexpr int kGrid = 16384;         // enough work to keep the graph multi-stream
constexpr unsigned kPoison = 0xDEAD;
constexpr unsigned kProduced = 0;

__global__ void FillKernel(unsigned* p, size_t n, unsigned v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    p[i] = v;
  }
}

// One fan-out step: kernels on the root stream feed a memcpy per side stream.
void CaptureStep(hipStream_t root, const std::vector<hipStream_t>& side, hipEvent_t fork,
                 const std::vector<hipEvent_t>& join, const std::vector<unsigned*>& src,
                 const std::vector<unsigned*>& dst, hipMemcpyKind kind, size_t bytes) {
  for (int i = 0; i < kBranches; ++i) {
    FillKernel<<<kGrid, kBlock, 0, root>>>(src[i], kElems, kProduced);
  }
  HIP_CHECK(hipEventRecord(fork, root));
  for (int i = 0; i < kBranches; ++i) {
    HIP_CHECK(hipStreamWaitEvent(side[i], fork, 0));
    HIP_CHECK(hipMemcpyAsync(dst[i], src[i], bytes, kind, side[i]));
    HIP_CHECK(hipEventRecord(join[i], side[i]));
    HIP_CHECK(hipStreamWaitEvent(root, join[i], 0));
  }
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Two chained fan-out steps, each a set of kernels feeding uncaptured memcpys that run on
 *    the SDMA engine. The second step's copies must observe their producer on every replay:
 *    the first step's copies leave a completion signal on the shared stream, and a copy that
 *    picks that up instead of its own producer's reads stale data.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphSdmaMemcpyOrdering.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_SdmaMemcpyOrderedAfterProducer) {
  // The collapse heuristic is read once at HIP init: it would otherwise fold the graph onto a
  // single in-order stream, where the queue itself provides the ordering. Set it in a child
  // process so it takes effect before any HIP call.
  hip::SpawnProc child(getSelfExePath());
  child.setEnv("DEBUG_HIP_GRAPH_MIN_OVERLAP", "0");
  REQUIRE(child.spawn("Unit_hipGraphLaunch_SdmaMemcpyOrderedAfterProducer_Child") == 0);
  REQUIRE(child.wait() == 0);
}

HIP_TEST_CASE(Unit_hipGraphLaunch_SdmaMemcpyOrderedAfterProducer_Child) {
  const hipMemcpyKind kCopyOnSdma = hipMemcpyDeviceToDeviceNoCU;
  const size_t bytes = kElems * sizeof(unsigned);

  hipStream_t root = nullptr;
  HIP_CHECK(hipStreamCreate(&root));
  hipEvent_t fork_a = nullptr, fork_b = nullptr;
  HIP_CHECK(hipEventCreateWithFlags(&fork_a, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&fork_b, hipEventDisableTiming));

  std::vector<hipStream_t> side(kBranches, nullptr);
  std::vector<hipEvent_t> join_a(kBranches, nullptr), join_b(kBranches, nullptr);
  std::vector<unsigned*> src_a(kBranches, nullptr), dst_a(kBranches, nullptr);
  std::vector<unsigned*> src_b(kBranches, nullptr), dst_b(kBranches, nullptr);
  for (int i = 0; i < kBranches; ++i) {
    HIP_CHECK(hipStreamCreateWithFlags(&side[i], hipStreamNonBlocking));
    HIP_CHECK(hipEventCreateWithFlags(&join_a[i], hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&join_b[i], hipEventDisableTiming));
    HIP_CHECK(hipMalloc(&src_a[i], bytes));
    HIP_CHECK(hipMalloc(&dst_a[i], bytes));
    HIP_CHECK(hipMalloc(&src_b[i], bytes));
    HIP_CHECK(hipMalloc(&dst_b[i], bytes));
  }

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamBeginCapture(root, hipStreamCaptureModeGlobal));
  CaptureStep(root, side, fork_a, join_a, src_a, dst_a, kCopyOnSdma, bytes);
  CaptureStep(root, side, fork_b, join_b, src_b, dst_b, kCopyOnSdma, bytes);
  HIP_CHECK(hipStreamEndCapture(root, &graph));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  for (int iter = 0; iter < 5; ++iter) {
    // Poison the sources: a copy that runs before its producer observes this instead.
    for (int i = 0; i < kBranches; ++i) {
      HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(src_a[i]), kPoison, kElems));
      HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(src_b[i]), kPoison, kElems));
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipGraphLaunch(graph_exec, 0));
    HIP_CHECK(hipDeviceSynchronize());

    // Sampling the ends is enough: the poison and the producer both cover the whole buffer.
    for (unsigned* buf : {dst_a[0], dst_b[0], dst_a[kBranches - 1], dst_b[kBranches - 1]}) {
      unsigned first = kPoison, last = kPoison;
      HIP_CHECK(hipMemcpy(&first, buf, sizeof(first), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&last, buf + kElems - 1, sizeof(last), hipMemcpyDeviceToHost));
      REQUIRE(first == kProduced);
      REQUIRE(last == kProduced);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  for (int i = 0; i < kBranches; ++i) {
    HIP_CHECK(hipFree(src_a[i]));
    HIP_CHECK(hipFree(dst_a[i]));
    HIP_CHECK(hipFree(src_b[i]));
    HIP_CHECK(hipFree(dst_b[i]));
    HIP_CHECK(hipEventDestroy(join_a[i]));
    HIP_CHECK(hipEventDestroy(join_b[i]));
    HIP_CHECK(hipStreamDestroy(side[i]));
  }
  HIP_CHECK(hipEventDestroy(fork_a));
  HIP_CHECK(hipEventDestroy(fork_b));
  HIP_CHECK(hipStreamDestroy(root));
}

/**
* End doxygen group GraphTest.
* @}
*/
