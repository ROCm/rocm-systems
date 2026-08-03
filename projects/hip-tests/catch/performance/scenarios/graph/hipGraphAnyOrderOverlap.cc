/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Correctness + performance tests for same-queue any-order overlap: under
// round-robin, the head barrier bit of later same-queue collisions is cleared so
// capable hardware can overlap them (non-head packets keep intra-segment order).
//
// The feature applies only under round-robin and is read once at instantiate, so
// force RR and compare feature ON vs OFF across separate processes:
//   DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=1 ./GraphPerformance "Performance_Graph_AnyOrderOverlap*"
//   DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=0 ./GraphPerformance "Performance_Graph_AnyOrderOverlap*"
//
// Mode 2 additionally clears cross-ring-only anchors in NON-oversubscribed levels
// (width <= pool). The *CrossRing tests below exercise that path; they need a pool
// >= graph width so no level is oversubscribed. The correctness case is width 4
// (fits the default pool of 4); the perf case is width 8, so force >= 8 queues:
//   DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 DEBUG_HIP_FORCE_GRAPH_QUEUES=8 \
//     DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=2 ./GraphPerformance "Performance_Graph_AnyOrderOverlap*"

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

// Ordering-token kernel: step i must observe val==expected before writing
// expected+1; a relaxed intra-chain dep reads a stale value and flags err.
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

// Relay step: must observe the producer's published token (src==1) before
// publishing its own (dst=1). Models a producer->consumer edge that carries a
// token; used to build cross-ring vs same-ring dependencies for mode 2.
__global__ void relayStep(const int* src, int* dst, int n, int* err, int burn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (src[i] != 1) atomicExch(&err[0], 1);
    float x = 1.0f;
    for (int k = 0; k < burn; ++k) x = x * 1.0001f + 0.0001f;
    if (x == 0.0f) atomicExch(&err[0], -1);  // keep burn live; never taken
    __threadfence_system();
    dst[i] = 1;
  }
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

// Correctness: independent oversubscribed chains; clearing a segment head must
// never relax an intra-chain dependency.
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

// Correctness: producer -> N children fork; a child colliding on the producer's
// queue must still observe its published token. Child chains are long enough to
// avoid single-stream collapse.
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

// Correctness for the mode-2 path: a NON-oversubscribed, cross-ring DAG.
// Level 1 nodes depend only on cross-ring producers (mode 2 clears their heads;
// ordering must survive via completion signals). Level 2 nodes depend on their
// same-ring producer (mode 2 must KEEP those heads; the barrier bit orders them).
// Run with pool >= width (default 4 == width) so no level is oversubscribed.
TEST_CASE("Performance_Graph_AnyOrderOverlap_CrossRing") {
  INFO("DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=" << FlagState());
  constexpr int W = 4;  // graph width; keep <= pool so levels are non-oversubscribed
  const int N = kGridBlocks * kBlock;

  std::vector<int*> buf0(W), buf1(W), buf2(W);
  for (int w = 0; w < W; ++w) {
    HIP_CHECK(hipMalloc(&buf0[w], N * sizeof(int)));
    HIP_CHECK(hipMalloc(&buf1[w], N * sizeof(int)));
    HIP_CHECK(hipMalloc(&buf2[w], N * sizeof(int)));
  }
  int* err = nullptr;
  HIP_CHECK(hipMalloc(&err, sizeof(int)));
  int nArg = N, burnArg = kBurn;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // L0: independent roots, each publishes buf0[w].
  std::vector<hipGraphNode_t> root(W);
  for (int w = 0; w < W; ++w)
    root[w] = AddKernel(graph, {}, reinterpret_cast<void*>(forkRoot), {&buf0[w], &nArg, &burnArg});
  // L1: cross[w] reads buf0[(w+1)%W] (cross-ring); 2 deps force per-node segments.
  std::vector<hipGraphNode_t> cross(W);
  for (int w = 0; w < W; ++w) {
    std::vector<hipGraphNode_t> deps = {root[(w + 1) % W], root[(w + 2) % W]};
    cross[w] = AddKernel(graph, deps, reinterpret_cast<void*>(relayStep),
                         {&buf0[(w + 1) % W], &buf1[w], &nArg, &err, &burnArg});
  }
  // L2: same[w] reads buf1[w] (same-ring producer cross[w]); cross[(w+2)] is a
  // cross-ring edge to keep per-node segmentation.
  for (int w = 0; w < W; ++w) {
    std::vector<hipGraphNode_t> deps = {cross[w], cross[(w + 2) % W]};
    AddKernel(graph, deps, reinterpret_cast<void*>(relayStep),
              {&buf1[w], &buf2[w], &nArg, &err, &burnArg});
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  int order_violations = 0;
  for (int r = 0; r < kReps; ++r) {
    for (int w = 0; w < W; ++w) {
      HIP_CHECK(hipMemsetAsync(buf0[w], 0, N * sizeof(int), stream));
      HIP_CHECK(hipMemsetAsync(buf1[w], 0, N * sizeof(int), stream));
      HIP_CHECK(hipMemsetAsync(buf2[w], 0, N * sizeof(int), stream));
    }
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
  for (int w = 0; w < W; ++w) {
    HIP_CHECK(hipFree(buf0[w]));
    HIP_CHECK(hipFree(buf1[w]));
    HIP_CHECK(hipFree(buf2[w]));
  }
}

// Correctness across graph update: updating a child head re-captures its packet
// (resetting the barrier bit); the head-barrier clear must be re-applied and the
// child must still observe the producer's token.
TEST_CASE("Performance_Graph_AnyOrderOverlap_Update") {
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
  std::vector<hipGraphNode_t> heads;
  for (int c = 0; c < kForkChildren; ++c) {
    hipGraphNode_t head =
        AddKernel(graph, {root}, reinterpret_cast<void*>(forkChild), {&tok, &nArg, &err});
    heads.push_back(head);
    hipGraphNode_t prev = head;
    for (int k = 1; k < kChildLen; ++k) {
      prev = AddKernel(graph, {prev}, reinterpret_cast<void*>(forkChild), {&tok, &nArg, &err});
    }
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Update every child head in place -> forces packet re-capture (barrier reset)
  // and exercises the head-barrier re-clear on the update path.
  for (auto head : heads) {
    std::vector<void*> args = {&tok, &nArg, &err};
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(forkChild);
    p.gridDim = dim3(kGridBlocks, 1, 1);
    p.blockDim = dim3(kBlock, 1, 1);
    p.sharedMemBytes = 0;
    p.kernelParams = args.data();
    p.extra = nullptr;
    HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, head, &p));
  }

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
  SECTION("relayStep detector") {
    int* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, N * sizeof(int)));
    int burnArg = 0;  // src==0 (never published), expects 1 -> must flag
    std::vector<void*> args = {&buf, &dst, &nArg, &err, &burnArg};
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(relayStep), dim3(kGridBlocks), dim3(kBlock),
                              args.data(), 0, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(dst));
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
// chain graph. Compare flag ON vs OFF across processes; no hard assert.
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

// Performance (informational): cross-ring, NON-oversubscribed DAG. Mode 1
// (positional) is inert here (nothing is oversubscribed), so this isolates mode 2's
// win: clearing cross-ring anchors lets levels pipeline across rings. Requires a
// pool >= width to stay non-oversubscribed, so run with DEBUG_HIP_FORCE_GRAPH_QUEUES=8
// (or more). Compare flag 0 vs 1 vs 2 across processes. No hard assert.
TEST_CASE("Performance_Graph_AnyOrderOverlap_CrossRingPerf") {
  INFO("DEBUG_HIP_GRAPH_ANYORDER_OVERLAP=" << FlagState());
  constexpr int W = 8;       // width; keep pool >= W (DEBUG_HIP_FORCE_GRAPH_QUEUES>=8)
  constexpr int L = 16;      // depth: room for cross-level overlap to accumulate
  constexpr int kTimedLaunches = 200;
  constexpr int kBusyIters = 8000;   // short kernels: barrier-wait latency (what the
  constexpr int kPerfBlocks = 1;     // clear removes) dominates, so overlap is visible
  const int N = kPerfBlocks * kBlock;

  std::vector<std::vector<float*>> buf(W, std::vector<float*>(L, nullptr));
  for (int w = 0; w < W; ++w)
    for (int l = 0; l < L; ++l) {
      HIP_CHECK(hipMalloc(&buf[w][l], N * sizeof(float)));
      HIP_CHECK(hipMemset(buf[w][l], 0, N * sizeof(float)));
    }
  int nArg = N, itersArg = kBusyIters;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<std::vector<hipGraphNode_t>> node(W, std::vector<hipGraphNode_t>(L));
  for (int w = 0; w < W; ++w)
    node[w][0] = AddKernel(graph, {}, reinterpret_cast<void*>(busy), {&buf[w][0], &nArg, &itersArg},
                           kPerfBlocks);
  // node[w][l] depends only on cross-ring producers from level l-1 (2 deps force
  // per-node segments); no same-ring edge => mode 2 clears every level anchor.
  for (int l = 1; l < L; ++l)
    for (int w = 0; w < W; ++w) {
      std::vector<hipGraphNode_t> deps = {node[(w + 1) % W][l - 1], node[(w + 2) % W][l - 1]};
      node[w][l] = AddKernel(graph, deps, reinterpret_cast<void*>(busy),
                             {&buf[w][l], &nArg, &itersArg}, kPerfBlocks);
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
  WARN("[AnyOrderOverlap-CrossRing] flag=" << FlagState() << " avg_launch=" << avg_us
       << " us over " << kTimedLaunches << " launches (width=" << W << ", levels=" << L << ")");

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  for (int w = 0; w < W; ++w)
    for (int l = 0; l < L; ++l) HIP_CHECK(hipFree(buf[w][l]));
}
