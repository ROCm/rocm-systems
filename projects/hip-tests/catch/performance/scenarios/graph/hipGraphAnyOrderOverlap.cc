/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Correctness + performance test for the same-queue any-order overlap
// optimization on oversubscribed graph queues.
//
// WHAT THE FEATURE DOES
//   When more parallel graph segments land on a queue than there are queues in
//   the pool, the runtime may CLEAR the AQL barrier bit on the *head* packet of
//   the later same-queue collisions so they overlap on capable HW instead of
//   serializing. Only the segment head is touched; every non-head packet keeps
//   barrier=1, so intra-segment order is preserved. The first segment on each
//   queue at each dependency level keeps barrier=1 and thus fences the whole
//   collision group behind the previous level's (dependency) work.
//
// THE FLAG (turn the feature ON/OFF)
//   DEBUG_HIP_GRAPH_ANYORDER_OVERLAP = 1  -> feature ON  (barrier bits cleared)
//   DEBUG_HIP_GRAPH_ANYORDER_OVERLAP = 0  -> feature OFF (default; fully ordered)
//   The flag is read ONCE at hipGraphInstantiate, so ON vs OFF must be compared
//   across SEPARATE processes:
//     DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=1 ./GraphPerformance "Performance_Graph_AnyOrderOverlap*"
//     DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=0 ./GraphPerformance "Performance_Graph_AnyOrderOverlap*"

#include <hip_test_common.hh>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kGridBlocks = 8;     // small grid -> GPU head-room so segments co-reside
constexpr int kBlock = 256;
constexpr int kBranches = 8;       // > default queue pool (4) => oversubscription
constexpr int kChainLen = 3;       // linear chain -> one segment per branch
constexpr int kForkChildren = 8;   // producer fans out to > pool children
constexpr int kChildLen = 5;       // child chains long enough to avoid collapse
constexpr int kReps = 200;         // many launches to surface nondeterministic races
constexpr int kBurn = 100000;      // widens any misordering window (does not touch tokens)

// Ordering-token kernel: step i of a chain must observe val==expected (its
// producer already ran) before writing expected+1. A relaxed intra-chain
// dependency makes a step read a stale value and record it in err.
__global__ void chainStep(int* val, int n, int expected, int* err, int burn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = 1.0f;
    for (int k = 0; k < burn; ++k) x = x * 1.0001f + 0.0001f;
    if (x == 0.0f) atomicExch(&err[0], -1);  // keep burn live; never taken
    if (val[i] != expected) atomicExch(&err[0], expected + 1);
    val[i] = expected + 1;
  }
}

// Fork producer: burn (widen window) then publish token = 1.
__global__ void forkRoot(int* tok, int n, int burn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = 1.0f;
    for (int k = 0; k < burn; ++k) x = x * 1.0001f + 0.0001f;
    if (x == 0.0f) tok[i] = -1;  // keep burn live; never taken
    __threadfence_system();
    tok[i] = 1;
  }
}

// Fork child head: must observe the producer's published token. Reading before
// the producer retired (a relaxed cross-segment same-queue dep) sets err.
__global__ void forkChild(int* tok, int n, int* err) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && tok[i] != 1) atomicExch(&err[0], 1);
}

// Plain busy kernel for the performance section.
__global__ void busy(float* d, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = d[i];
    for (int k = 0; k < iters; ++k) x = x * 1.0001f + 0.0001f;
    d[i] = x;
  }
}

hipGraphNode_t AddKernel(hipGraph_t graph, std::vector<hipGraphNode_t> deps, void* func,
                         std::vector<void*> args, int gridBlocks = kGridBlocks) {
  hipKernelNodeParams p{};
  p.func = func;
  p.gridDim = dim3(gridBlocks, 1, 1);
  p.blockDim = dim3(kBlock, 1, 1);
  p.sharedMemBytes = 0;
  p.kernelParams = args.data();
  p.extra = nullptr;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps.empty() ? nullptr : deps.data(), deps.size(),
                                  &p));
  return node;
}

const char* FlagState() {
  const char* env = std::getenv("DEBUG_HIP_GRAPH_ANYORDER_OVERLAP");
  return env ? env : "(default/OFF)";
}

}  // namespace

// Correctness: independent oversubscribed chains. Validates that clearing a
// segment head never relaxes an intra-segment (intra-chain) dependency.
TEST_CASE("Performance_Graph_AnyOrderOverlap_Chains") {
  INFO("DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=" << FlagState());
  const int N = kGridBlocks * kBlock;

  std::vector<int*> val(kBranches, nullptr);
  for (int b = 0; b < kBranches; ++b) HIP_CHECK(hipMalloc(&val[b], N * sizeof(int)));
  int* err = nullptr;
  HIP_CHECK(hipMalloc(&err, sizeof(int)));

  std::vector<int> expected(static_cast<size_t>(kBranches) * kChainLen);
  int nArg = N, burnArg = kBurn;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  for (int b = 0; b < kBranches; ++b) {
    hipGraphNode_t prev = nullptr;
    for (int k = 0; k < kChainLen; ++k) {
      const int idx = b * kChainLen + k;
      expected[idx] = k;
      std::vector<void*> args = {&val[b], &nArg, &expected[idx], &err, &burnArg};
      std::vector<hipGraphNode_t> deps;
      if (prev != nullptr) deps.push_back(prev);
      prev = AddKernel(graph, deps, reinterpret_cast<void*>(chainStep), args);
    }
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  int order_violations = 0, value_violations = 0;
  std::vector<int> host_val(N);
  for (int r = 0; r < kReps; ++r) {
    for (int b = 0; b < kBranches; ++b) HIP_CHECK(hipMemsetAsync(val[b], 0, N * sizeof(int), stream));
    HIP_CHECK(hipMemsetAsync(err, 0, sizeof(int), stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int host_err = 0;
    HIP_CHECK(hipMemcpy(&host_err, err, sizeof(int), hipMemcpyDeviceToHost));
    if (host_err != 0) ++order_violations;
    for (int b = 0; b < kBranches; ++b) {
      HIP_CHECK(hipMemcpy(host_val.data(), val[b], N * sizeof(int), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i) {
        if (host_val[i] != kChainLen) { ++value_violations; break; }
      }
    }
  }
  INFO("ordering_violations=" << order_violations << " value_violations=" << value_violations);
  REQUIRE(order_violations == 0);
  REQUIRE(value_violations == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(err));
  for (auto* p : val) HIP_CHECK(hipFree(p));
}

// Correctness: producer -> N children fork (cross-segment same-queue dependency).
// Round-robin lands a child on the producer's queue as a non-first collision, so
// its head barrier is a clear candidate. The child must still observe the
// producer's published token. Child chains are long enough that the graph does
// NOT collapse to a single stream at default flags.
TEST_CASE("Performance_Graph_AnyOrderOverlap_Fork") {
  INFO("DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=" << FlagState());
  const int N = kGridBlocks * kBlock;

  int* tok = nullptr;
  int* err = nullptr;
  HIP_CHECK(hipMalloc(&tok, N * sizeof(int)));
  HIP_CHECK(hipMalloc(&err, sizeof(int)));
  int nArg = N, burnArg = kBurn;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t root =
      AddKernel(graph, {}, reinterpret_cast<void*>(forkRoot), {&tok, &nArg, &burnArg});
  for (int c = 0; c < kForkChildren; ++c) {
    // Head node checks the producer's token; the rest extend the chain so the
    // child segment carries enough work to keep the graph multi-stream.
    hipGraphNode_t head =
        AddKernel(graph, {root}, reinterpret_cast<void*>(forkChild), {&tok, &nArg, &err});
    hipGraphNode_t prev = head;
    for (int k = 1; k < kChildLen; ++k) {
      prev = AddKernel(graph, {prev}, reinterpret_cast<void*>(forkChild), {&tok, &nArg, &err});
    }
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  int order_violations = 0;
  for (int r = 0; r < kReps; ++r) {
    HIP_CHECK(hipMemsetAsync(tok, 0, N * sizeof(int), stream));
    HIP_CHECK(hipMemsetAsync(err, 0, sizeof(int), stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int host_err = 0;
    HIP_CHECK(hipMemcpy(&host_err, err, sizeof(int), hipMemcpyDeviceToHost));
    if (host_err != 0) ++order_violations;
  }
  INFO("ordering_violations=" << order_violations);
  REQUIRE(order_violations == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(err));
  HIP_CHECK(hipFree(tok));
}

// Detector self-test: prove the ordering checks above actually fire on a known
// bad state, so a PASS in the correctness cases cannot be a false negative.
TEST_CASE("Performance_Graph_AnyOrderOverlap_DetectorSelfTest") {
  const int N = kGridBlocks * kBlock;
  int* buf = nullptr;
  int* err = nullptr;
  HIP_CHECK(hipMalloc(&buf, N * sizeof(int)));
  HIP_CHECK(hipMalloc(&err, sizeof(int)));
  HIP_CHECK(hipMemset(buf, 0, N * sizeof(int)));  // token/value never published
  HIP_CHECK(hipMemset(err, 0, sizeof(int)));

  int nArg = N;
  SECTION("chainStep detector") {
    int expectedArg = 1, burnArg = 0;  // expects 1, observes 0 -> must flag
    std::vector<void*> args = {&buf, &nArg, &expectedArg, &err, &burnArg};
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(chainStep), dim3(kGridBlocks), dim3(kBlock),
                              args.data(), 0, nullptr));
  }
  SECTION("forkChild detector") {
    std::vector<void*> args = {&buf, &nArg, &err};
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(forkChild), dim3(kGridBlocks), dim3(kBlock),
                              args.data(), 0, nullptr));
  }
  HIP_CHECK(hipDeviceSynchronize());

  int host_err = 0;
  HIP_CHECK(hipMemcpy(&host_err, err, sizeof(int), hipMemcpyDeviceToHost));
  INFO("detector err=" << host_err << " (expected non-zero)");
  REQUIRE(host_err != 0);

  HIP_CHECK(hipFree(err));
  HIP_CHECK(hipFree(buf));
}

// Performance (informational): steady-state launch time for the oversubscribed
// chain graph. Compare across two runs with the flag ON vs OFF (separate
// processes). No hard assert -- timing is machine/load dependent.
TEST_CASE("Performance_Graph_AnyOrderOverlap_Perf") {
  INFO("DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=" << FlagState());
  constexpr int kTimedLaunches = 200;
  constexpr int kBusyIters = 500000;
  const int N = kGridBlocks * kBlock;

  std::vector<float*> bufs(kBranches, nullptr);
  for (int b = 0; b < kBranches; ++b) {
    HIP_CHECK(hipMalloc(&bufs[b], N * sizeof(float)));
    HIP_CHECK(hipMemset(bufs[b], 0, N * sizeof(float)));
  }
  int nArg = N, itersArg = kBusyIters;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  for (int b = 0; b < kBranches; ++b) {
    hipGraphNode_t prev = nullptr;
    for (int k = 0; k < kChainLen; ++k) {
      std::vector<void*> args = {&bufs[b], &nArg, &itersArg};
      std::vector<hipGraphNode_t> deps;
      if (prev != nullptr) deps.push_back(prev);
      prev = AddKernel(graph, deps, reinterpret_cast<void*>(busy), args);
    }
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  for (int i = 0; i < 5; ++i) HIP_CHECK(hipGraphLaunch(exec, stream));  // warmup
  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t start{}, stop{};
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < kTimedLaunches; ++i) HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  const double avg_us = static_cast<double>(ms) * 1000.0 / kTimedLaunches;
  WARN("[AnyOrderOverlap] flag=" << FlagState() << " avg_launch=" << avg_us
       << " us over " << kTimedLaunches << " launches (branches=" << kBranches
       << ", chain_len=" << kChainLen << ", grid_blocks=" << kGridBlocks << ")");

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  for (auto* p : bufs) HIP_CHECK(hipFree(p));
}
