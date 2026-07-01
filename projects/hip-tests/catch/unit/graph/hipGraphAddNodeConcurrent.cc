/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios :
 Regression for ROCM-24799.

 The fix promotes GraphNode::nextID and Graph::nextID from plain
 `static int` to std::atomic<int>, so the read-modify-write inside
 the constructors is indivisible. Without the fix, concurrent graph
 construction across multiple threads races on the shared counter and
 produces duplicate id_ values; intra-graph duplicates then truncate
 Graph::FindPathsDFS and silently drop nodes from the dispatched
 packet.

 Tests in this file:
   - ConcurrentConstruct_NextIDRace: per-thread graph, asserts every
     GraphNode::id_ is unique via hipGraphDebugDotPrint readback.
   - ConcurrentCreate_GraphIDRace:   companion for Graph::nextID,
     asserts every Graph::id_ is unique via the same channel.
   - ConcurrentConstruct_LaunchDropSmoke: per-thread linear-chain of
     memset nodes; launches and checks for any dropped node via
     sentinel buffer (the customer's exact symptom path).
*/
#include <hip_test_common.hh>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kThreads = 4;
constexpr int kNodesPerThread = 256;

void BuildAndDumpGraph(std::atomic<int>* start_gate, std::string dot_path,
                       std::vector<int>* ids_out) {
  HIP_CHECK_THREAD(hipSetDevice(0));

  hipGraph_t graph{};
  HIP_CHECK_THREAD(hipGraphCreate(&graph, 0));

  // Release the gate so every thread enters the AddNode loop at the same
  // moment, maximising concurrent nextID++ contention.
  start_gate->fetch_add(1, std::memory_order_release);
  while (start_gate->load(std::memory_order_acquire) < kThreads) {
    std::this_thread::yield();
  }

  for (int i = 0; i < kNodesPerThread; ++i) {
    hipGraphNode_t n{};
    HIP_CHECK_THREAD(hipGraphAddEmptyNode(&n, graph, nullptr, 0));
  }

  HIP_CHECK_THREAD(hipGraphDebugDotPrint(graph, dot_path.c_str(), 0));

  // hipGraphDebugDotPrint emits one "graph_<G>_node_<id>"[ declaration
  // line per GraphNode. <id> is GraphNode::id_; if two nodes share id_,
  // two matching declarations land in the file.
  std::ifstream in(dot_path);
  std::string line;
  std::regex decl_re(R"(\"graph_\d+_node_(\d+)\"\[)");
  while (std::getline(in, line)) {
    auto begin = std::sregex_iterator(line.begin(), line.end(), decl_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      ids_out->push_back(std::stoi((*it)[1].str()));
    }
  }
  in.close();
  std::remove(dot_path.c_str());

  HIP_CHECK_THREAD(hipGraphDestroy(graph));
}

}  // namespace

/**
 Companion for the second fixed counter: Graph::nextID. Concurrent
 hipGraphCreate calls race on the global counter; pre-fix two Graph
 objects can be assigned the same id_. Detection uses
 hipGraphDebugDotPrint's "subgraph cluster_<id>" token. Threads race
 in a tight hipGraphCreate burst (phase 1), then a single thread dumps
 each graph and extracts the cluster id (phase 2), and global
 uniqueness is asserted.
*/
HIP_TEST_CASE(Unit_hipGraph_ConcurrentCreate_GraphIDRace) {
  constexpr int kGraphThreads = 4;
  constexpr int kGraphsPerThread = 256;

  std::atomic<int> start_gate{0};
  std::vector<std::vector<hipGraph_t>> per_thread_graphs(kGraphThreads);

  auto worker = [&](int t) {
    HIP_CHECK_THREAD(hipSetDevice(0));
    start_gate.fetch_add(1, std::memory_order_release);
    while (start_gate.load(std::memory_order_acquire) < kGraphThreads) {
      std::this_thread::yield();
    }
    per_thread_graphs[t].reserve(kGraphsPerThread);
    for (int i = 0; i < kGraphsPerThread; ++i) {
      hipGraph_t g{};
      HIP_CHECK_THREAD(hipGraphCreate(&g, 0));
      per_thread_graphs[t].push_back(g);
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(kGraphThreads);
  for (int t = 0; t < kGraphThreads; ++t) workers.emplace_back(worker, t);
  for (auto& w : workers) w.join();
  HIP_CHECK_THREAD_FINALIZE();

  std::regex cluster_re(R"(subgraph cluster_(\d+))");
  std::unordered_map<int, int> count_across;
  const std::string dot_path =
      (std::filesystem::temp_directory_path() /
       "hipGraphAddNodeConcurrent_graphid.dot").string();
  for (int t = 0; t < kGraphThreads; ++t) {
    for (hipGraph_t g : per_thread_graphs[t]) {
      HIP_CHECK(hipGraphDebugDotPrint(g, dot_path.c_str(), 0));
      std::ifstream in(dot_path);
      std::string line;
      while (std::getline(in, line)) {
        std::smatch m;
        if (std::regex_search(line, m, cluster_re)) {
          ++count_across[std::stoi(m[1].str())];
          break;
        }
      }
      in.close();
      std::remove(dot_path.c_str());
      HIP_CHECK(hipGraphDestroy(g));
    }
  }

  REQUIRE(static_cast<int>(count_across.size()) == kGraphThreads * kGraphsPerThread);
  for (auto& kv : count_across) {
    REQUIRE(kv.second == 1);
  }
}

HIP_TEST_CASE(Unit_hipGraph_ConcurrentConstruct_NextIDRace) {
  std::atomic<int> start_gate{0};
  std::vector<std::vector<int>> per_thread_ids(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  const auto tmp_dir = std::filesystem::temp_directory_path();
  for (int t = 0; t < kThreads; ++t) {
    std::string path =
        (tmp_dir / ("hipGraphAddNodeConcurrent_dot_" + std::to_string(t) + ".txt")).string();
    workers.emplace_back(BuildAndDumpGraph, &start_gate, std::move(path),
                         &per_thread_ids[t]);
  }
  for (auto& w : workers) w.join();
  HIP_CHECK_THREAD_FINALIZE();

  // Aggregate id counts across all threads. Any count > 1 is a
  // duplicate id (intra- or inter-graph) and direct evidence of the
  // race on nextID.
  std::unordered_map<int, int> count_across;
  for (int t = 0; t < kThreads; ++t) {
    REQUIRE(static_cast<int>(per_thread_ids[t].size()) == kNodesPerThread);
    for (int id : per_thread_ids[t]) {
      ++count_across[id];
    }
  }
  for (auto& kv : count_across) {
    REQUIRE(kv.second == 1);
  }
}

/**
 Smoke test that reproduces the customer-visible symptom: each thread
 builds its own graph as a linear chain of memset nodes (chain stays
 on the segment-scheduling path, avoiding the complex-graph fallback
 to classic that would mask drops). Each memset writes a unique
 non-sentinel value into a slot of a sentinel-filled device buffer.
 After launch, any slot still holding the sentinel was dropped by
 FindPathsDFS due to a duplicate id earlier in the chain. Independent
 of hipGraphDebugDotPrint output format.
*/
HIP_TEST_CASE(Unit_hipGraph_ConcurrentConstruct_LaunchDropSmoke) {
  constexpr uint32_t kSentinel = 0xDEADBEEFu;
  // Quick level keeps the test under a second; full run amps contention
  // up to a config that catches the race ~100% in a single invocation.
  const int kSmokeThreads = isQuickLevel() ? 4 : 32;
  const int kSmokeChainLen = isQuickLevel() ? 128 : 512;
  const int kSmokeIters = isQuickLevel() ? 5 : 30;
  std::atomic<int> start_gate{0};
  std::atomic<int> total_dropped{0};

  auto worker = [&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    const size_t bytes = sizeof(uint32_t) * kSmokeChainLen;
    uint32_t* d_buf = nullptr;
    HIP_CHECK_THREAD(hipMalloc(&d_buf, bytes));
    std::vector<uint32_t> sentinel(kSmokeChainLen, kSentinel);

    for (int iter = 0; iter < kSmokeIters; ++iter) {
      HIP_CHECK_THREAD(hipMemcpy(d_buf, sentinel.data(), bytes, hipMemcpyHostToDevice));

      hipGraph_t graph{};
      HIP_CHECK_THREAD(hipGraphCreate(&graph, 0));

      // Release the gate once per thread, before the first iter's build
      // loop, to maximise concurrent overlap on nextID++.
      if (iter == 0) {
        start_gate.fetch_add(1, std::memory_order_release);
        while (start_gate.load(std::memory_order_acquire) < kSmokeThreads) {
          std::this_thread::yield();
        }
      }

      // Build a linear chain: memset[i] depends on memset[i-1].
      hipGraphNode_t prev{};
      for (int i = 0; i < kSmokeChainLen; ++i) {
        hipMemsetParams p{};
        p.dst = d_buf + i;
        p.pitch = 0;
        p.value = static_cast<uint32_t>(i + 1);  // != kSentinel
        p.elementSize = sizeof(uint32_t);
        p.width = 1;
        p.height = 1;
        hipGraphNode_t n{};
        if (i == 0) {
          HIP_CHECK_THREAD(hipGraphAddMemsetNode(&n, graph, nullptr, 0, &p));
        } else {
          HIP_CHECK_THREAD(hipGraphAddMemsetNode(&n, graph, &prev, 1, &p));
        }
        prev = n;
      }

      hipGraphExec_t exec{};
      HIP_CHECK_THREAD(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      HIP_CHECK_THREAD(hipGraphLaunch(exec, 0));
      HIP_CHECK_THREAD(hipDeviceSynchronize());

      std::vector<uint32_t> host(kSmokeChainLen);
      HIP_CHECK_THREAD(hipMemcpy(host.data(), d_buf, bytes, hipMemcpyDeviceToHost));
      int dropped = 0;
      for (int i = 0; i < kSmokeChainLen; ++i) {
        if (host[i] == kSentinel) ++dropped;
      }
      total_dropped.fetch_add(dropped, std::memory_order_relaxed);

      HIP_CHECK_THREAD(hipGraphExecDestroy(exec));
      HIP_CHECK_THREAD(hipGraphDestroy(graph));
    }
    HIP_CHECK_THREAD(hipFree(d_buf));
  };

  std::vector<std::thread> workers2;
  workers2.reserve(kSmokeThreads);
  for (int t = 0; t < kSmokeThreads; ++t) workers2.emplace_back(worker);
  for (auto& w : workers2) w.join();
  HIP_CHECK_THREAD_FINALIZE();

  REQUIRE(total_dropped.load() == 0);
}
