/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphSegmentOrdering hipGraphSegmentOrdering
 * @{
 * @ingroup GraphTest
 * Ordering checks for the segmented graph executor. Each kernel node verifies on the GPU that
 * the nodes it must follow have already finished in the same launch, and root nodes verify that
 * the previous launch has finished, so a dropped or misplaced wait is counted as a violation
 * rather than surfacing only as a timing-dependent wrong result. Kernels spin for different
 * lengths so that a node released too early runs ahead of a still-busy dependency.
 *
 * Multi-stream graphs here are deeper than four dependency levels so the default scheduler
 * spreads them across streams instead of collapsing small graphs onto the launch stream.
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Mirrors the executor choice in hipGraphInstantiate: the classic executor (PAL backend, the
// Windows default, or DEBUG_HIP_GRAPH_CLASSIC_PATH) schedules nodes differently, and these
// tests target the segmented executor's stream assignment and sync plan.
bool UsesSegmentedGraphPath() {
#if HT_WIN
  return false;
#else
  auto flag_enabled = [](const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && (std::strcmp(value, "true") == 0 || std::atoi(value) != 0);
  };
  return !flag_enabled("GPU_ENABLE_PAL") && !flag_enabled("DEBUG_HIP_GRAPH_CLASSIC_PATH");
#endif
}

constexpr int kMaxChecks = 16;

// What a node verifies when it starts. Graph edges are added separately; checks name only
// nodes that stay enabled, so they remain valid while other nodes are disabled.
struct NodeChecks {
  // Node whose done count gives the launch index: the node itself, or an ancestor that has
  // already finished the current launch.
  int launch_ref;
  int num_deps;
  int deps[kMaxChecks];  // must have finished the current launch
  int num_prev_launch;
  int prev_launch[kMaxChecks];  // must have finished the previous launch
  unsigned long long spin_ticks;
  // Optional: every element of verify_data must equal *verify_expected.
  const int* verify_data;
  const int* verify_expected;
  int verify_count;
};

__device__ unsigned LoadDoneCount(const unsigned* done_counts, int node) {
  return __hip_atomic_load(&done_counts[node], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

__global__ void CheckedStep(unsigned* done_counts, const NodeChecks* all_checks, int self,
                            unsigned* violations) {
  const NodeChecks& checks = all_checks[self];
  __shared__ unsigned num_violations;
  if (threadIdx.x == 0) {
    const unsigned long long start = wall_clock64();
    const unsigned ref_done = LoadDoneCount(done_counts, checks.launch_ref);
    const unsigned launch_index = checks.launch_ref == self ? ref_done : ref_done - 1;
    num_violations = 0;
    for (int dep_idx = 0; dep_idx < checks.num_deps; ++dep_idx) {
      if (LoadDoneCount(done_counts, checks.deps[dep_idx]) < launch_index + 1) ++num_violations;
    }
    for (int prev_idx = 0; prev_idx < checks.num_prev_launch; ++prev_idx) {
      if (LoadDoneCount(done_counts, checks.prev_launch[prev_idx]) < launch_index) {
        ++num_violations;
      }
    }
    while (wall_clock64() - start < checks.spin_ticks) {
    }
  }
  __syncthreads();
  if (checks.verify_data != nullptr) {
    const int expected = *checks.verify_expected;
    for (int elem = threadIdx.x; elem < checks.verify_count; elem += blockDim.x) {
      if (checks.verify_data[elem] != expected) {
        atomicAdd(&num_violations, 1u);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (num_violations != 0) atomicAdd(violations, num_violations);
    __hip_atomic_fetch_add(&done_counts[self], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// A graph of CheckedStep kernel nodes. Node ids must be added in topological order.
class CheckedGraph {
 public:
  explicit CheckedGraph(int num_nodes)
      : checks_(num_nodes), graph_deps_(num_nodes), nodes_(num_nodes) {
    for (int node = 0; node < num_nodes; ++node) checks_[node].launch_ref = node;
    HIP_CHECK(hipMalloc(&done_counts_, num_nodes * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&violations_, sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&device_checks_, num_nodes * sizeof(NodeChecks)));
    HIP_CHECK(hipMemset(done_counts_, 0, num_nodes * sizeof(unsigned)));
    HIP_CHECK(hipMemset(violations_, 0, sizeof(unsigned)));
  }

  // Runs during unwinding when a REQUIRE fails, so cleanup errors are ignored instead of thrown.
  ~CheckedGraph() {
    if (exec_ != nullptr) static_cast<void>(hipGraphExecDestroy(exec_));
    if (graph_ != nullptr) static_cast<void>(hipGraphDestroy(graph_));
    static_cast<void>(hipFree(done_counts_));
    static_cast<void>(hipFree(violations_));
    static_cast<void>(hipFree(device_checks_));
  }

  int NumNodes() const { return static_cast<int>(checks_.size()); }
  NodeChecks& Checks(int node) { return checks_[node]; }

  // Graph edge dep -> node that is also verified at run time.
  void AddCheckedEdge(int node, int dep) {
    AddEdge(node, dep);
    AddCheck(node, dep);
  }
  void AddEdge(int node, int dep) { graph_deps_[node].push_back(dep); }
  void AddCheck(int node, int dep) {
    NodeChecks& checks = checks_[node];
    REQUIRE(checks.num_deps < kMaxChecks);
    checks.deps[checks.num_deps++] = dep;
  }
  void AddPrevLaunchCheck(int node, int prev_launch_node) {
    NodeChecks& checks = checks_[node];
    REQUIRE(checks.num_prev_launch < kMaxChecks);
    checks.prev_launch[checks.num_prev_launch++] = prev_launch_node;
  }

  void Instantiate() {
    HIP_CHECK(hipMemcpy(device_checks_, checks_.data(), checks_.size() * sizeof(NodeChecks),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphCreate(&graph_, 0));
    for (int node = 0; node < NumNodes(); ++node) {
      int self = node;
      void* args[] = {&done_counts_, &device_checks_, &self, &violations_};
      hipKernelNodeParams params{};
      params.func = reinterpret_cast<void*>(CheckedStep);
      params.gridDim = dim3(1);
      params.blockDim = dim3(256);
      params.kernelParams = args;
      std::vector<hipGraphNode_t> deps;
      for (int dep : graph_deps_[node]) deps.push_back(nodes_[dep]);
      HIP_CHECK(hipGraphAddKernelNode(&nodes_[node], graph_, deps.data(), deps.size(), &params));
    }
    HIP_CHECK(hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
  }

  void SetEnabled(int node, bool enabled) {
    HIP_CHECK(hipGraphNodeSetEnabled(exec_, nodes_[node], enabled ? 1 : 0));
  }

  hipGraphExec_t Exec() const { return exec_; }

  unsigned Violations() const {
    unsigned host_violations = 0;
    HIP_CHECK(hipMemcpy(&host_violations, violations_, sizeof(unsigned), hipMemcpyDeviceToHost));
    return host_violations;
  }

  std::vector<unsigned> DoneCounts() const {
    std::vector<unsigned> host_done(checks_.size());
    HIP_CHECK(hipMemcpy(host_done.data(), done_counts_, host_done.size() * sizeof(unsigned),
                        hipMemcpyDeviceToHost));
    return host_done;
  }

 private:
  std::vector<NodeChecks> checks_;
  std::vector<std::vector<int>> graph_deps_;
  std::vector<hipGraphNode_t> nodes_;
  unsigned* done_counts_ = nullptr;
  unsigned* violations_ = nullptr;
  NodeChecks* device_checks_ = nullptr;
  hipGraph_t graph_ = nullptr;
  hipGraphExec_t exec_ = nullptr;
};

unsigned long long WallClockTicksPerUs() {
  int device = 0;
  int wall_clock_khz = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, device));
  return std::max(1ULL, static_cast<unsigned long long>(wall_clock_khz) / 1000);
}

constexpr int kButterflyStages = 6;

int ButterflyNode(int width, int stage, int lane) { return stage * width + lane; }

// kButterflyStages stages of `width` nodes; node (stage, lane) depends on lanes `lane` and
// `lane + 1` of the previous stage. Every node is both a fork and a join, so each forms its own
// segment and its stage is its dependency level. With `width` above the graph stream pool
// (DEBUG_HIP_FORCE_GRAPH_QUEUES, default 4), same-level segments share streams; at stage 0
// those peers have no dependencies at all. Stage-0 nodes check that the previous launch's last
// stage finished.
void BuildButterfly(CheckedGraph& graph, int width) {
  const unsigned long long ticks_per_us = WallClockTicksPerUs();
  const int last_stage = kButterflyStages - 1;
  for (int stage = 0; stage < kButterflyStages; ++stage) {
    for (int lane = 0; lane < width; ++lane) {
      const int node = ButterflyNode(width, stage, lane);
      // Alternate long and short nodes so peers sharing a stream finish at different times.
      graph.Checks(node).spin_ticks = ((stage + lane) % 2 == 0 ? 20 : 5) * ticks_per_us;
      if (stage == 0) {
        for (int prev_lane = 0; prev_lane < width; ++prev_lane) {
          graph.AddPrevLaunchCheck(node, ButterflyNode(width, last_stage, prev_lane));
        }
        continue;
      }
      graph.AddCheckedEdge(node, ButterflyNode(width, stage - 1, lane));
      graph.AddCheckedEdge(node, ButterflyNode(width, stage - 1, (lane + 1) % width));
    }
  }
}

void RequireAllDoneCounts(const CheckedGraph& graph, unsigned expected) {
  const std::vector<unsigned> done_counts = graph.DoneCounts();
  for (int node = 0; node < graph.NumNodes(); ++node) {
    INFO("node " << node);
    REQUIRE(done_counts[node] == expected);
  }
}

__global__ void FillWithSeed(int* data, int count, int* expected, int seed) {
  for (int elem = blockIdx.x * blockDim.x + threadIdx.x; elem < count;
       elem += gridDim.x * blockDim.x) {
    data[elem] = seed;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *expected = seed;
}

__global__ void ChainStep(unsigned* sequence, int self, int chain_len, unsigned* violations) {
  const unsigned position =
      __hip_atomic_load(sequence, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  if (position % chain_len != static_cast<unsigned>(self)) atomicAdd(violations, 1u);
  __hip_atomic_store(sequence, position + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Repeated launches of a butterfly graph wider than the stream pool, so independent segments
 *    of one dependency level share a stream. Verifies that every node starts only after its
 *    dependencies finished, that a launch starts only after the previous one finished, and that
 *    no node is skipped.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphSegmentOrdering.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphSegmentOrdering_SameLevelPeers) {
  if (!UsesSegmentedGraphPath()) HIP_SKIP_TEST("Requires the segmented graph executor");
  constexpr unsigned kLaunches = 50;
  const int width = GENERATE(5, 8);
  INFO("width " << width);

  CheckedGraph graph(width * kButterflyStages);
  BuildButterfly(graph, width);
  graph.Instantiate();

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  for (unsigned launch = 0; launch < kLaunches; ++launch) {
    HIP_CHECK(hipGraphLaunch(graph.Exec(), stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(graph.Violations() == 0);
  RequireAllDoneCounts(graph, kLaunches);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Before each launch of a butterfly graph, the launch stream rewrites a buffer with a new
 *    value (kernel, memset or host-to-device copy). Every stage-0 node, most of which run on
 *    other streams, must read the new value, and the next rewrite must not overtake the graph.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphSegmentOrdering.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphSegmentOrdering_ForkSeesLaunchStreamWrites) {
  if (!UsesSegmentedGraphPath()) HIP_SKIP_TEST("Requires the segmented graph executor");
  constexpr int kWidth = 8;
  constexpr int kIterations = 20;
  constexpr int kVerifyCount = 1 << 16;

  int* verify_data = nullptr;
  int* verify_expected = nullptr;
  HIP_CHECK(hipMalloc(&verify_data, kVerifyCount * sizeof(int)));
  HIP_CHECK(hipMalloc(&verify_expected, sizeof(int)));
  HIP_CHECK(hipMemset(verify_data, 0, kVerifyCount * sizeof(int)));
  HIP_CHECK(hipMemset(verify_expected, 0, sizeof(int)));

  CheckedGraph graph(kWidth * kButterflyStages);
  BuildButterfly(graph, kWidth);
  for (int lane = 0; lane < kWidth; ++lane) {
    NodeChecks& checks = graph.Checks(ButterflyNode(kWidth, 0, lane));
    checks.verify_data = verify_data;
    checks.verify_expected = verify_expected;
    checks.verify_count = kVerifyCount;
  }
  graph.Instantiate();

  // Per-iteration host sources for the copy variant: a source must not be rewritten while an
  // earlier asynchronous copy may still read it.
  int* host_sources = nullptr;
  HIP_CHECK(hipHostMalloc(&host_sources, kIterations * (kVerifyCount + 1) * sizeof(int)));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  SECTION("kernel") {
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      FillWithSeed<<<64, 256, 0, stream>>>(verify_data, kVerifyCount, verify_expected,
                                           iteration + 1);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipGraphLaunch(graph.Exec(), stream));
    }
  }
  SECTION("memset") {
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(verify_data), iteration + 1,
                                  kVerifyCount, stream));
      HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(verify_expected),
                                  iteration + 1, 1, stream));
      HIP_CHECK(hipGraphLaunch(graph.Exec(), stream));
    }
  }
  SECTION("host to device copy") {
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      int* host_source = host_sources + iteration * (kVerifyCount + 1);
      for (int elem = 0; elem <= kVerifyCount; ++elem) host_source[elem] = iteration + 1;
      HIP_CHECK(hipMemcpyAsync(verify_data, host_source, kVerifyCount * sizeof(int),
                               hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipMemcpyAsync(verify_expected, host_source + kVerifyCount, sizeof(int),
                               hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipGraphLaunch(graph.Exec(), stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(graph.Violations() == 0);
  RequireAllDoneCounts(graph, kIterations);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(host_sources));
  HIP_CHECK(hipFree(verify_data));
  HIP_CHECK(hipFree(verify_expected));
}

/**
 * Test Description
 * ------------------------
 *  - A chain of three diamonds: fork -> {P, Q} -> join chain C0 -> C1 -> C2, where C2 forks the
 *    next diamond. The join chain shares a stream with one of P and Q and waits on the other.
 *    Disabling C0 (and C1) removes the packet at the head of that segment; the remaining
 *    nodes must still start only after both P and Q finished, including after re-enabling.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphSegmentOrdering.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphSegmentOrdering_DisabledJoinHead) {
  if (!UsesSegmentedGraphPath()) HIP_SKIP_TEST("Requires the segmented graph executor");
  constexpr int kDiamonds = 3;
  constexpr int kNodesPerDiamond = 5;
  constexpr unsigned kLaunchesPerConfig = 10;
  const unsigned long long ticks_per_us = WallClockTicksPerUs();

  // Node 0 is the root; diamond k occupies ids 1 + k * kNodesPerDiamond onward.
  auto diamond_node = [](int diamond, int offset) {
    return 1 + diamond * kNodesPerDiamond + offset;
  };
  enum DiamondOffset { kP = 0, kQ, kC0, kC1, kC2 };
  const int last_node = diamond_node(kDiamonds - 1, kC2);

  CheckedGraph graph(1 + kDiamonds * kNodesPerDiamond);
  graph.AddPrevLaunchCheck(0, last_node);
  for (int diamond = 0; diamond < kDiamonds; ++diamond) {
    const int fork_node = diamond == 0 ? 0 : diamond_node(diamond - 1, kC2);
    const int p_node = diamond_node(diamond, kP);
    const int q_node = diamond_node(diamond, kQ);
    // Alternate which branch is slow so the slow one is the cross-stream dependency of the
    // join chain in at least one diamond, whichever stream each branch is assigned.
    const bool p_slow = diamond % 2 == 0;
    for (int branch_node : {p_node, q_node}) {
      graph.AddCheckedEdge(branch_node, fork_node);
      graph.Checks(branch_node).launch_ref = 0;
    }
    graph.Checks(p_node).spin_ticks = (p_slow ? 80 : 10) * ticks_per_us;
    graph.Checks(q_node).spin_ticks = (p_slow ? 10 : 80) * ticks_per_us;

    graph.AddEdge(diamond_node(diamond, kC0), p_node);
    graph.AddEdge(diamond_node(diamond, kC0), q_node);
    graph.AddEdge(diamond_node(diamond, kC1), diamond_node(diamond, kC0));
    graph.AddEdge(diamond_node(diamond, kC2), diamond_node(diamond, kC1));
    for (int chain_offset : {kC0, kC1, kC2}) {
      const int chain_node = diamond_node(diamond, chain_offset);
      graph.Checks(chain_node).launch_ref = 0;
      graph.AddCheck(chain_node, p_node);
      graph.AddCheck(chain_node, q_node);
      graph.Checks(chain_node).spin_ticks = 2 * ticks_per_us;
    }
  }
  graph.Instantiate();

  struct Config {
    const char* name;
    std::vector<int> disabled_offsets;  // applied to every diamond unless one_diamond >= 0
    int one_diamond;
  };
  const std::vector<Config> configs = {
      {"all enabled", {}, -1},
      {"C0 disabled", {kC0}, -1},
      {"C0 and C1 disabled", {kC0, kC1}, -1},
      {"re-enabled", {}, -1},
      {"C0 disabled in middle diamond", {kC0}, 1},
      {"re-enabled again", {}, -1},
  };

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::vector<unsigned> expected_done(graph.NumNodes(), 0);
  for (const Config& config : configs) {
    INFO("config: " << config.name);
    std::vector<bool> enabled(graph.NumNodes(), true);
    for (int diamond = 0; diamond < kDiamonds; ++diamond) {
      if (config.one_diamond >= 0 && diamond != config.one_diamond) continue;
      for (int offset : config.disabled_offsets) enabled[diamond_node(diamond, offset)] = false;
    }
    for (int node = 0; node < graph.NumNodes(); ++node) {
      graph.SetEnabled(node, enabled[node]);
      if (enabled[node]) expected_done[node] += kLaunchesPerConfig;
    }

    for (unsigned launch = 0; launch < kLaunchesPerConfig; ++launch) {
      HIP_CHECK(hipGraphLaunch(graph.Exec(), stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(graph.Violations() == 0);
    const std::vector<unsigned> done_counts = graph.DoneCounts();
    for (int node = 0; node < graph.NumNodes(); ++node) {
      INFO("node " << node);
      REQUIRE(done_counts[node] == expected_done[node]);
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Single-stream chains whose lengths straddle the boundaries of the chunks a large packet
 *    batch is submitted in. Every kernel checks it runs exactly in chain order, and no packet
 *    may be lost or repeated across launches.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphSegmentOrdering.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphSegmentOrdering_LongChainBatches) {
  if (!UsesSegmentedGraphPath()) HIP_SKIP_TEST("Requires the segmented graph executor");
  constexpr unsigned kLaunches = 3;
  const int chain_len = GENERATE(1, 8, 9, 24, 25, 56, 57, 504, 505, 1000);
  INFO("chain length " << chain_len);

  unsigned* sequence = nullptr;
  unsigned* violations = nullptr;
  HIP_CHECK(hipMalloc(&sequence, sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&violations, sizeof(unsigned)));
  HIP_CHECK(hipMemset(sequence, 0, sizeof(unsigned)));
  HIP_CHECK(hipMemset(violations, 0, sizeof(unsigned)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t prev_node = nullptr;
  for (int position = 0; position < chain_len; ++position) {
    int self = position;
    int len = chain_len;
    void* args[] = {&sequence, &self, &len, &violations};
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(ChainStep);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, prev_node ? &prev_node : nullptr,
                                    prev_node ? 1 : 0, &params));
    prev_node = node;
  }
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  for (unsigned launch = 0; launch < kLaunches; ++launch) {
    HIP_CHECK(hipGraphLaunch(exec, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  unsigned host_sequence = 0;
  unsigned host_violations = 0;
  HIP_CHECK(hipMemcpy(&host_sequence, sequence, sizeof(unsigned), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&host_violations, violations, sizeof(unsigned), hipMemcpyDeviceToHost));
  REQUIRE(host_violations == 0);
  REQUIRE(host_sequence == kLaunches * chain_len);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(sequence));
  HIP_CHECK(hipFree(violations));
}

/**
 * Test Description
 * ------------------------
 *  - Graph with a join-and-fork EMPTY node (E0) whose successors land on a different stream.
 *    The EMPTY-only segment must emit a completion signal so the downstream barrier does not
 *    block forever. Reproduces the hang reported in ROCM-32125. Also covers a leaf EMPTY node
 *    (E1) that forks to the two output kernels.
 *
 *    Shape (K* are kernel nodes, E* are hipGraphAddEmptyNode nodes):
 *      K0 -> K1 -> E0 <- K0   (redundant K0->E0 makes E0 a join)
 *      E0 -> G0 -> G1 -> E1
 *      E1 -> P, F
 *
 *    Run with DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=2 to force multi-stream assignment even on
 *    shallow graphs where the default scheduler would collapse to one stream.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphSegmentOrdering.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphSegmentOrdering_EmptyNodeCrossStreamCompletion) {
  if (!UsesSegmentedGraphPath()) HIP_SKIP_TEST("Requires the segmented graph executor");

  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));
  unsigned* violations = nullptr;
  HIP_CHECK(hipMalloc(&violations, sizeof(unsigned)));
  HIP_CHECK(hipMemset(violations, 0, sizeof(unsigned)));

  // Use ChainStep as a lightweight sequence checker for this topology test.
  // Each kernel writes its own sequence slot; we just need all 100 launches to complete.
  unsigned* sequence = nullptr;
  HIP_CHECK(hipMalloc(&sequence, sizeof(unsigned)));
  HIP_CHECK(hipMemset(sequence, 0, sizeof(unsigned)));

  // Build the graph: K0->K1->E0<-K0, E0->G0->G1->E1->P, E1->F
  hipGraphNode_t nK0, nK1, nG0, nG1, nP, nF, nE0, nE1;
  constexpr int kChainLen = 1;
  int self0 = 0, len = kChainLen;
  void* args0[] = {&sequence, &self0, &len, &violations};
  hipKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(ChainStep);
  kp.gridDim = dim3(1); kp.blockDim = dim3(1);
  kp.kernelParams = args0;

  HIP_CHECK(hipGraphAddKernelNode(&nK0, g, nullptr, 0, &kp));
  HIP_CHECK(hipGraphAddKernelNode(&nK1, g, &nK0, 1, &kp));
  hipGraphNode_t e0_deps[] = {nK0, nK1};
  HIP_CHECK(hipGraphAddEmptyNode(&nE0, g, e0_deps, 2));
  HIP_CHECK(hipGraphAddKernelNode(&nG0, g, &nE0, 1, &kp));
  HIP_CHECK(hipGraphAddKernelNode(&nG1, g, &nG0, 1, &kp));
  HIP_CHECK(hipGraphAddEmptyNode(&nE1, g, &nG1, 1));
  HIP_CHECK(hipGraphAddKernelNode(&nP, g, &nE1, 1, &kp));
  HIP_CHECK(hipGraphAddKernelNode(&nF, g, &nE1, 1, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr unsigned kLaunches = 100;
  for (unsigned i = 0; i < kLaunches; ++i) {
    HIP_CHECK(hipMemsetAsync(sequence, 0, sizeof(unsigned), stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
  }
  // If E0 or E1's completion signal is never emitted this synchronize hangs forever.
  HIP_CHECK(hipStreamSynchronize(stream));

  unsigned host_violations = 0;
  HIP_CHECK(hipMemcpy(&host_violations, violations, sizeof(unsigned), hipMemcpyDeviceToHost));
  REQUIRE(host_violations == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(g));
  HIP_CHECK(hipFree(sequence));
  HIP_CHECK(hipFree(violations));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
