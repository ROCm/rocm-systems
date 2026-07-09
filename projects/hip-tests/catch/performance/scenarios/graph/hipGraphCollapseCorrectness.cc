/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// End-to-end data-correctness test for the single-stream collapse heuristic.
//
// The collapse heuristic (DEBUG_HIP_GRAPH_MIN_OVERLAP) can silently fold a
// multi-stream graph onto one in-order stream. This test builds a diamond graph
// with real cross-branch data dependencies and verifies the output is correct
// regardless of whether collapse fires, so a dropped/mis-ordered dependency
// would corrupt the result and fail the test (not just check liveness).
//
// To cover both code paths, run the binary twice:
//   DEBUG_HIP_GRAPH_MIN_OVERLAP=0 ./GraphPerformance "Performance_Graph_CollapseCorrectness"
//   DEBUG_HIP_GRAPH_MIN_OVERLAP=2 ./GraphPerformance "Performance_Graph_CollapseCorrectness"
// (the flag is read once at hipGraphInstantiate, hence separate processes).

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kN = 4096;          // small kernels so the gate may collapse them
constexpr int kBlock = 256;
constexpr int kIters = 200;       // many launches to surface nondeterministic races
constexpr int kPoison = -1;       // a stale (pre-dependency) read yields this

__global__ void k_set(int* a, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = val;
}

__global__ void k_mul(const int* in, int* out, int factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * factor;
}

__global__ void k_add(const int* x, const int* y, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i];
}

hipGraphNode_t AddKernel(hipGraph_t graph, std::vector<hipGraphNode_t> deps,
                         void* func, std::vector<void*> args) {
  hipKernelNodeParams p{};
  p.func = func;
  p.gridDim = dim3((kN + kBlock - 1) / kBlock, 1, 1);
  p.blockDim = dim3(kBlock, 1, 1);
  p.sharedMemBytes = 0;
  p.kernelParams = args.data();
  p.extra = nullptr;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps.empty() ? nullptr : deps.data(),
                                  deps.size(), &p));
  return node;
}

}  // namespace

/**
 * Diamond DAG:  a=set(1) -> b=a*2 (branch 1)
 *                        -> c=a*3 (branch 2)
 *               out = b + c  (join, depends on both branches)
 * Correct result is out[i] == 5 for all i. Buffers are poisoned before each
 * launch, so any node running before its producer reads kPoison and fails.
 */
TEST_CASE("Performance_Graph_CollapseCorrectness") {
  const char* env = std::getenv("DEBUG_HIP_GRAPH_MIN_OVERLAP");
  INFO("DEBUG_HIP_GRAPH_MIN_OVERLAP=" << (env ? env : "(default)"));

  const size_t bytes = static_cast<size_t>(kN) * sizeof(int);
  int *a = nullptr, *b = nullptr, *c = nullptr, *out = nullptr;
  HIP_CHECK(hipMalloc(&a, bytes));
  HIP_CHECK(hipMalloc(&b, bytes));
  HIP_CHECK(hipMalloc(&c, bytes));
  HIP_CHECK(hipMalloc(&out, bytes));

  int one = 1, two = 2, three = 3, n = kN;
  void* a_p = a;
  void* b_p = b;
  void* c_p = c;
  void* out_p = out;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t n_set =
      AddKernel(graph, {}, reinterpret_cast<void*>(k_set), {&a_p, &one, &n});
  hipGraphNode_t n_b = AddKernel(graph, {n_set}, reinterpret_cast<void*>(k_mul),
                                 {&a_p, &b_p, &two, &n});
  hipGraphNode_t n_c = AddKernel(graph, {n_set}, reinterpret_cast<void*>(k_mul),
                                 {&a_p, &c_p, &three, &n});
  AddKernel(graph, {n_b, n_c}, reinterpret_cast<void*>(k_add),
            {&b_p, &c_p, &out_p, &n});

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<int> host(kN);
  for (int iter = 0; iter < kIters; ++iter) {
    // Poison every buffer so a stale read (wrong order) is observable.
    HIP_CHECK(hipMemsetAsync(a, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(b, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(c, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(out, 0xFF, bytes, stream));

    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipMemcpyAsync(host.data(), out, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int bad = -1;
    for (int i = 0; i < kN; ++i) {
      if (host[i] != 5) { bad = i; break; }
    }
    INFO("iter=" << iter << " first_bad_index=" << bad
                 << " value=" << (bad >= 0 ? host[bad] : 5)
                 << " (poison=" << kPoison << ")");
    REQUIRE(bad == -1);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
  HIP_CHECK(hipFree(out));
}
