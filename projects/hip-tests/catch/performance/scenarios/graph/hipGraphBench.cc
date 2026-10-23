/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @addtogroup GraphBenchPerformance GraphBenchPerformance
 * @{
 * @ingroup PerformanceTestGraph
 * hipGraphLaunch submission latency across graph topologies of ~N kernel nodes,
 * plus an ordering-correctness verifier for the same topologies.
 *
 * Topologies (all use approximately N kernel nodes):
 *   straight  one linear chain of N nodes (1 segment)
 *   paths2    lead (N/4) -> 2 parallel branches (N/4 each) -> tail (N/4)
 *   paths4    lead (N/6) -> 4 parallel branches (N/6 each) -> tail (N/6)
 *   full2     2 independent chains of N/2 nodes, no join
 *   full4     4 independent chains of N/4 nodes, no join
 *
 * Knobs (environment variables):
 *   GRAPH_BENCH_TOPOLOGY         straight|paths2|paths4|full2|full4|all  (default: all)
 *   GRAPH_BENCH_SIZE             total kernel nodes for Launch/Verify     (default: 1024)
 *   GRAPH_BENCH_ITERS            timed launches per measurement           (default: 1000,
 *                                                                          100 at level_0)
 *   GRAPH_BENCH_MODE             no-sync|sync|both                        (default: no-sync)
 *                                  no-sync = submission only, sync = submission + GPU
 *   GRAPH_BENCH_VERIFY_ITERS     launches per topology in Verify          (default: 50)
 *   GRAPH_BENCH_VERIFY_DELAY_US  per-node busy-wait widening the race     (default: 1)
 *
 * Example:
 *   GRAPH_BENCH_TOPOLOGY=straight GRAPH_BENCH_MODE=both \
 *     ./GraphPerformance "Performance_GraphBench_Sweep"
 */

namespace {

__global__ void null_kernel() {}

// Writes buf[node_id] = sum(buf[deps]) + 1. Dependencies are read at entry and
// the result is written after delay_cycles, so a successor scheduled before this
// node finishes reads a stale 0 and the deficit propagates to the exit nodes.
__global__ void verify_kernel(int* buf, int node_id, int dep0, int dep1, int dep2, int dep3,
                              int num_deps, long long delay_cycles) {
  int value = 1;
  if (num_deps > 0) value += buf[dep0];
  if (num_deps > 1) value += buf[dep1];
  if (num_deps > 2) value += buf[dep2];
  if (num_deps > 3) value += buf[dep3];

  if (delay_cycles > 0) {
    const long long start = clock64();
    while (clock64() - start < delay_cycles) {
    }
  }

  buf[node_id] = value;
}

// Widest join is paths4's, with 4 predecessors.
constexpr int kMaxDeps = 4;

// Present only when building a graph for verification. Tracks node IDs, the
// CPU-computed expected value of every node, and stable kernel-arg storage.
struct VerifyContext {
  int* dev_buf = nullptr;
  int next_id = 0;
  long long delay_cycles = 0;

  std::vector<int> expected;
  std::vector<int> exits;
  std::unordered_map<hipGraphNode_t, int> node_to_id;

  // {node_id, dep0, dep1, dep2, dep3, num_deps}. Kernel params point into this
  // storage, so it is reserved to the exact node count before the build.
  std::vector<std::array<int, 6>> node_args;
};

// Adds one kernel node depending on deps[0..num_deps-1]. Returns the verify
// node ID, or -1 when no verify context is supplied.
int AddNode(hipGraph_t graph, hipGraphNode_t* node, const hipGraphNode_t* deps, int num_deps,
            VerifyContext* verify) {
  hipKernelNodeParams params{};
  params.gridDim = dim3(1, 1, 1);
  params.blockDim = dim3(1, 1, 1);

  if (verify == nullptr) {
    params.func = reinterpret_cast<void*>(null_kernel);
    HIP_CHECK(hipGraphAddKernelNode(node, graph, deps, num_deps, &params));
    return -1;
  }

  std::array<int, kMaxDeps> dep_ids{};
  int expected_value = 1;
  for (int dep_idx = 0; dep_idx < num_deps; ++dep_idx) {
    dep_ids[dep_idx] = verify->node_to_id.at(deps[dep_idx]);
    expected_value += verify->expected[dep_ids[dep_idx]];
  }

  const int node_id = verify->next_id++;
  verify->expected.push_back(expected_value);
  auto& node_args = verify->node_args.emplace_back(
      std::array<int, 6>{node_id, dep_ids[0], dep_ids[1], dep_ids[2], dep_ids[3], num_deps});

  void* args[] = {&verify->dev_buf, &node_args[0], &node_args[1], &node_args[2],
                  &node_args[3],    &node_args[4], &node_args[5], &verify->delay_cycles};
  params.func = reinterpret_cast<void*>(verify_kernel);
  params.kernelParams = args;
  HIP_CHECK(hipGraphAddKernelNode(node, graph, deps, num_deps, &params));

  verify->node_to_id[*node] = node_id;
  return node_id;
}

// Appends a chain of chain_length nodes. The first node depends on *root when
// root is non-null. Returns the tail handle and stores the tail's verify ID.
hipGraphNode_t AddChain(hipGraph_t graph, const hipGraphNode_t* root, int chain_length,
                        VerifyContext* verify, int* tail_id) {
  hipGraphNode_t prev{};
  hipGraphNode_t node{};
  for (int i = 0; i < chain_length; ++i) {
    const hipGraphNode_t* deps = i == 0 ? root : &prev;
    *tail_id = AddNode(graph, &node, deps, deps == nullptr ? 0 : 1, verify);
    prev = node;
  }
  return prev;
}

hipGraph_t CreateStraight(int size, VerifyContext* verify) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int tail_id = -1;
  AddChain(graph, nullptr, size, verify, &tail_id);
  if (verify != nullptr) verify->exits.push_back(tail_id);
  return graph;
}

// lead chain -> num_paths parallel branches -> join -> tail chain.
// The join's expected value is the sum of all branch tails plus one, so a
// branch finishing after the join shows up as a deficit at the single exit.
hipGraph_t CreateMultiPath(int size, int num_paths, VerifyContext* verify) {
  const int segment_length = std::max(1, size / (num_paths + 2));
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int tail_id = -1;
  const hipGraphNode_t split = AddChain(graph, nullptr, segment_length, verify, &tail_id);

  std::vector<hipGraphNode_t> path_ends(num_paths);
  for (int path = 0; path < num_paths; ++path) {
    path_ends[path] = AddChain(graph, &split, segment_length, verify, &tail_id);
  }

  hipGraphNode_t join{};
  tail_id = AddNode(graph, &join, path_ends.data(), num_paths, verify);
  AddChain(graph, &join, segment_length - 1, verify, &tail_id);
  if (verify != nullptr) verify->exits.push_back(tail_id);
  return graph;
}

// num_paths independent chains with no synchronization point between them.
hipGraph_t CreateFullParallel(int size, int num_paths, VerifyContext* verify) {
  const int segment_length = std::max(1, size / num_paths);
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  for (int path = 0; path < num_paths; ++path) {
    int tail_id = -1;
    AddChain(graph, nullptr, segment_length, verify, &tail_id);
    if (verify != nullptr) verify->exits.push_back(tail_id);
  }
  return graph;
}

struct Topology {
  const char* name;
  hipGraph_t (*create)(int size, VerifyContext* verify);
  // Exact node count the builder emits; can exceed size for small sizes
  // because every segment has at least one node.
  int (*node_count)(int size);
};

const Topology kTopologies[] = {
    {"straight", CreateStraight, [](int size) { return size; }},
    {"paths2", [](int size, VerifyContext* verify) { return CreateMultiPath(size, 2, verify); },
     [](int size) { return std::max(1, size / 4) * 4; }},
    {"paths4", [](int size, VerifyContext* verify) { return CreateMultiPath(size, 4, verify); },
     [](int size) { return std::max(1, size / 6) * 6; }},
    {"full2", [](int size, VerifyContext* verify) { return CreateFullParallel(size, 2, verify); },
     [](int size) { return std::max(1, size / 2) * 2; }},
    {"full4", [](int size, VerifyContext* verify) { return CreateFullParallel(size, 4, verify); },
     [](int size) { return std::max(1, size / 4) * 4; }},
};

const char* EnvString(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : fallback;
}

int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::atoi(value) : fallback;
}

std::vector<const Topology*> SelectedTopologies() {
  const std::string requested = EnvString("GRAPH_BENCH_TOPOLOGY", "all");
  std::vector<const Topology*> selected;
  for (const auto& topology : kTopologies) {
    if (requested == "all" || requested == topology.name) selected.push_back(&topology);
  }
  INFO("Unknown GRAPH_BENCH_TOPOLOGY '" << requested
                                        << "'; expected straight|paths2|paths4|full2|full4|all");
  REQUIRE_FALSE(selected.empty());
  return selected;
}

// Timing modes to run: false = submission only, true = submission + GPU.
std::vector<bool> SelectedSyncModes() {
  const std::string mode = EnvString("GRAPH_BENCH_MODE", "no-sync");
  if (mode == "both") return {false, true};
  INFO("Unknown GRAPH_BENCH_MODE '" << mode << "'; expected no-sync|sync|both");
  REQUIRE((mode == "no-sync" || mode == "sync"));
  return {mode == "sync"};
}

int TimedIters() { return EnvInt("GRAPH_BENCH_ITERS", isQuickLevel() ? 100 : 1000); }

const char* ModeName(bool sync_in_timing) {
  return sync_in_timing ? "sync (submission+GPU)" : "no-sync (submission only)";
}

double ElapsedUs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
      .count();
}

// Cold-instantiates graph and returns the elapsed time; caller owns *exec.
double InstantiateUs(hipGraph_t graph, hipGraphExec_t* exec) {
  const auto start = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphInstantiate(exec, graph, nullptr, nullptr, 0));
  return ElapsedUs(start);
}

// Average hipGraphLaunch latency over iters launches after 10 warmup launches.
double LaunchUs(hipGraphExec_t exec, int iters, bool sync_in_timing) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  for (int i = 0; i < 10; ++i) HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  double total_us = 0.0;
  for (int i = 0; i < iters; ++i) {
    const auto start = std::chrono::steady_clock::now();
    HIP_CHECK(hipGraphLaunch(exec, stream));
    if (sync_in_timing) HIP_CHECK(hipStreamSynchronize(stream));
    total_us += ElapsedUs(start);
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  return total_us / iters;
}

struct Measurement {
  double instantiate_us;
  double launch_us;
};

Measurement Measure(const Topology& topology, int size, int iters, bool sync_in_timing) {
  hipGraph_t graph = topology.create(size, nullptr);
  hipGraphExec_t exec;
  const double instantiate_us = InstantiateUs(graph, &exec);
  HIP_CHECK(hipGraphDestroy(graph));
  const double launch_us = LaunchUs(exec, iters, sync_in_timing);
  HIP_CHECK(hipGraphExecDestroy(exec));
  return {instantiate_us, launch_us};
}

void PrintHeader(const char* mode, int iters) {
  hipDeviceProp_t props;
  int device_id;
  HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));
  CONSOLE_PRINT("\nDevice : %s", props.name);
  CONSOLE_PRINT("Mode   : %s", mode);
  CONSOLE_PRINT("Iters  : %d per measurement\n", iters);
}

std::string SweepHeaderRow(const std::vector<const Topology*>& topologies) {
  char cell[32];
  std::string row = "size   ";
  for (const auto* topology : topologies) {
    std::snprintf(cell, sizeof(cell), "  %10s", topology->name);
    row += cell;
  }
  return row + "\n" + std::string(7 + topologies.size() * 12, '-');
}

struct VerifyResult {
  int num_nodes;
  int num_exits;
  int expected_exit_value;  // identical for every exit by construction
  int min_exit_value;       // smallest exit value observed over all launches
  int bad_iters;            // launches with at least one wrong node value
  long long bad_values;     // wrong node values summed over all launches
  double launch_us;         // average launch + synchronize time
};

// Builds the graph with verify_kernel nodes, launches it verify_iters times and
// compares every node's value with the CPU-computed expectation.
VerifyResult VerifyTopology(const Topology& topology, int size, long long delay_cycles,
                            int verify_iters) {
  const int num_nodes = topology.node_count(size);
  VerifyContext verify;
  verify.delay_cycles = delay_cycles;
  verify.node_args.reserve(num_nodes);
  verify.expected.reserve(num_nodes);
  HIP_CHECK(hipMalloc(&verify.dev_buf, num_nodes * sizeof(int)));

  hipGraph_t graph = topology.create(size, &verify);
  REQUIRE(verify.next_id == num_nodes);
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));

  VerifyResult result{};
  result.num_nodes = num_nodes;
  result.num_exits = static_cast<int>(verify.exits.size());
  result.expected_exit_value = verify.expected[verify.exits.front()];
  result.min_exit_value = result.expected_exit_value;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<int> host(num_nodes);
  int num_reported = 0;
  double total_launch_us = 0.0;
  for (int iter = 0; iter < verify_iters; ++iter) {
    HIP_CHECK(hipMemset(verify.dev_buf, 0, num_nodes * sizeof(int)));
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto start = std::chrono::steady_clock::now();
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    total_launch_us += ElapsedUs(start);
    HIP_CHECK(hipMemcpy(host.data(), verify.dev_buf, num_nodes * sizeof(int),
                        hipMemcpyDeviceToHost));

    for (int exit_id : verify.exits) {
      result.min_exit_value = std::min(result.min_exit_value, host[exit_id]);
    }
    int bad_values_this_iter = 0;
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
      if (host[node_id] == verify.expected[node_id]) continue;
      if (num_reported++ < 8) {
        CONSOLE_PRINT("  [%s] iter %d node %d: got %d, expected %d (ordering violation)",
                      topology.name, iter, node_id, host[node_id], verify.expected[node_id]);
      }
      ++bad_values_this_iter;
    }
    result.bad_values += bad_values_this_iter;
    result.bad_iters += bad_values_this_iter > 0;
  }
  result.launch_us = total_launch_us / verify_iters;

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipFree(verify.dev_buf));
  return result;
}

}  // namespace

/**
 * Cold instantiation time and average launch latency for each selected
 * topology at GRAPH_BENCH_SIZE nodes, per timing mode.
 */
HIP_TEST_CASE(Performance_GraphBench_Launch) {
  const auto topologies = SelectedTopologies();
  const int size = EnvInt("GRAPH_BENCH_SIZE", 1024);
  const int iters = TimedIters();

  for (bool sync_in_timing : SelectedSyncModes()) {
    PrintHeader(ModeName(sync_in_timing), iters);
    CONSOLE_PRINT("Size   : %d nodes\n", size);
    CONSOLE_PRINT("%-10s  %12s  %12s", "topology", "inst (us)", "launch (us)");
    CONSOLE_PRINT("%s", std::string(38, '-').c_str());
    for (const auto* topology : topologies) {
      const Measurement result = Measure(*topology, size, iters, sync_in_timing);
      CONSOLE_PRINT("%-10s  %9.3f     %9.3f", topology->name, result.instantiate_us,
                    result.launch_us);
    }
  }
}

/**
 * Sweep sizes 1, 2, 4, ..., 8192 for each selected topology and print an
 * instantiation table and a launch-latency table per timing mode.
 */
HIP_TEST_CASE(Performance_GraphBench_Sweep) {
  const auto topologies = SelectedTopologies();
  const int iters = TimedIters();
  std::vector<int> sizes;
  for (int size = 1; size <= 8192; size *= 2) sizes.push_back(size);

  for (bool sync_in_timing : SelectedSyncModes()) {
    std::vector<std::vector<Measurement>> results(sizes.size());
    for (size_t size_idx = 0; size_idx < sizes.size(); ++size_idx) {
      for (const auto* topology : topologies) {
        results[size_idx].push_back(Measure(*topology, sizes[size_idx], iters, sync_in_timing));
      }
    }

    PrintHeader(ModeName(sync_in_timing), iters);
    for (bool launch_table : {false, true}) {
      CONSOLE_PRINT("--- %s (us) ---", launch_table ? "launch" : "instantiate");
      CONSOLE_PRINT("%s", SweepHeaderRow(topologies).c_str());
      for (size_t size_idx = 0; size_idx < sizes.size(); ++size_idx) {
        char cell[32];
        std::snprintf(cell, sizeof(cell), "%-7d", sizes[size_idx]);
        std::string row = cell;
        for (const auto& result : results[size_idx]) {
          std::snprintf(cell, sizeof(cell), "  %9.3f us",
                        launch_table ? result.launch_us : result.instantiate_us);
          row += cell;
        }
        CONSOLE_PRINT("%s", row.c_str());
      }
      CONSOLE_PRINT("");
    }
  }
}

/**
 * Ordering check: every node runs an ordering-sensitive reduction kernel and all
 * node values are compared with CPU expectations after each launch.
 */
HIP_TEST_CASE(Performance_GraphBench_Verify) {
  const auto topologies = SelectedTopologies();
  const int size = EnvInt("GRAPH_BENCH_SIZE", 1024);
  const int verify_iters = EnvInt("GRAPH_BENCH_VERIFY_ITERS", 50);
  const int verify_delay_us = EnvInt("GRAPH_BENCH_VERIFY_DELAY_US", 1);

  hipDeviceProp_t props;
  int device_id;
  HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));
  // clockRate is in kHz; fall back to ~1.7 GHz if the runtime reports 0.
  const long long cycles_per_us = props.clockRate > 0 ? props.clockRate / 1000 : 1700;
  const long long delay_cycles = static_cast<long long>(verify_delay_us) * cycles_per_us;

  CONSOLE_PRINT("\nDevice : %s", props.name);
  CONSOLE_PRINT("Mode   : verify (ordering check, size=%d, iters=%d, delay=%dus/node)\n", size,
                verify_iters, verify_delay_us);
  CONSOLE_PRINT("%-10s  %6s  %5s  %10s  %10s  %9s  %10s  %12s", "topology", "nodes", "exits",
                "exit want", "exit min", "bad iters", "bad values", "launch (us)");
  CONSOLE_PRINT("%s", std::string(88, '-').c_str());

  long long total_bad_values = 0;
  for (const auto* topology : topologies) {
    const VerifyResult result = VerifyTopology(*topology, size, delay_cycles, verify_iters);
    CONSOLE_PRINT("%-10s  %6d  %5d  %10d  %10d  %4d/%-4d  %10lld  %12.3f", topology->name,
                  result.num_nodes, result.num_exits, result.expected_exit_value,
                  result.min_exit_value, result.bad_iters, verify_iters, result.bad_values,
                  result.launch_us);
    total_bad_values += result.bad_values;
  }
  REQUIRE(total_bad_values == 0);
}

/**
 * End doxygen group GraphBenchPerformance.
 * @}
 */
