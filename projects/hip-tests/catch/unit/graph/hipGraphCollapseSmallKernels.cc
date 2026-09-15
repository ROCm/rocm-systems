/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphCollapseSmallKernels hipGraphCollapseSmallKernels
 * @{
 * @ingroup GraphTest
 * Verifies that the barrier-ROI collapse heuristic (DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0)
 * folds a graph of tiny independent kernels onto a single stream. The dot print is
 * parsed after instantiation: all nodes must carry the same StreamId.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <vector>
#if HT_WIN
#include <process.h>
#else
#include <unistd.h>
#endif

// 1-thread trivial kernel — intentionally tiny so the ROI heuristic fires.
__global__ void tiny_write(int* out, int val) { *out = val; }

// Parse all "Stream: <N>" segment-assignment values from a dot file dumped by
// DEBUG_HIP_GRAPH_DOT_PRINT=2. These reflect the segment stream assignment
// (not the HW stream ID), so collapsed graphs show all Stream: 0 while
// multi-stream graphs show varied values.
static std::set<int> ParseSegmentStreams(const std::string& path) {
  std::ifstream f(path);
  std::set<int> ids;
  std::string line;
  std::regex re("Stream: (\\d+)");
  while (std::getline(f, line)) {
    std::sregex_iterator it(line.begin(), line.end(), re), end;
    for (; it != end; ++it) {
      ids.insert(std::stoi((*it)[1].str()));
    }
  }
  return ids;
}

// Build a fan-out graph: one root kernel whose output is read by N leaf kernels.
// N tiny independent leaves guarantees multiple segments → collapse is meaningful.
static void BuildFanOutGraph(hipGraph_t graph, int* d_root, std::vector<int*>& d_leaves,
                             int num_leaves) {
  // Root node: writes 1 to d_root
  hipKernelNodeParams root_params{};
  int root_val = 1;
  void* root_args[] = {&d_root, &root_val};
  root_params.func = reinterpret_cast<void*>(tiny_write);
  root_params.gridDim = dim3(1);
  root_params.blockDim = dim3(1);
  root_params.kernelParams = reinterpret_cast<void**>(root_args);

  hipGraphNode_t root_node;
  HIP_CHECK(hipGraphAddKernelNode(&root_node, graph, nullptr, 0, &root_params));

  // Leaf nodes: each writes its index to its own output buffer, depends on root
  for (int i = 0; i < num_leaves; ++i) {
    hipKernelNodeParams leaf_params{};
    int leaf_val = i + 10;
    void* leaf_args[] = {&d_leaves[i], &leaf_val};
    leaf_params.func = reinterpret_cast<void*>(tiny_write);
    leaf_params.gridDim = dim3(1);
    leaf_params.blockDim = dim3(1);
    leaf_params.kernelParams = reinterpret_cast<void**>(leaf_args);

    hipGraphNode_t leaf_node;
    HIP_CHECK(hipGraphAddKernelNode(&leaf_node, graph, &root_node, 1, &leaf_params));
  }
}

/**
 * Test: with mode=0 (collapse-eligible), all graph nodes collapse to a single stream.
 * Verified by parsing StreamId entries from the dot print output.
 */
// Helper: return the dot file path dumped by DEBUG_HIP_GRAPH_DOT_PRINT=2 and
// remove any stale copy so the runtime writes a fresh one on the next launch.
static std::string PrepareDotFile() {
#if HT_WIN
  int pid = _getpid();
#else
  pid_t pid = getpid();
#endif
  std::string path = "graph_" + std::to_string(pid) + "_dot_print_launch_1";
  std::remove(path.c_str());  // delete stale file from a prior test case
  return path;
}

// RAII guard: delete the generated dot file when the test scope exits. Runs even
// if a REQUIRE assertion throws, so the build tree is never left with a stale dot
// file on failure. DEBUG_HIP_GRAPH_DOT_PRINT is provided by the ctest add_test env
// (see CMakeLists.txt), so the test itself no longer sets/clears it.
struct DotFileGuard {
  std::string path;
  explicit DotFileGuard(const std::string& p) : path(p) {}
  ~DotFileGuard() { std::remove(path.c_str()); }
};

TEST_CASE("Unit_hipGraph_CollapseSmallKernels_SingleStream", "[graph][collapse][level_2]") {
  // Requires: DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0 DEBUG_HIP_GRAPH_MIN_OVERLAP=2
  //           DEBUG_HIP_GRAPH_DOT_PRINT=2
  // Flags are read once at HIP init so must be set before process launch;
  // the ctest add_test wiring in CMakeLists.txt provides them.
  // Verifies that a fan-out graph of tiny kernels collapses to a single stream.
  constexpr int kNumLeaves = 8;

  int* d_root;
  HIP_CHECK(hipMalloc(&d_root, sizeof(int)));
  std::vector<int*> d_leaves(kNumLeaves);
  for (int i = 0; i < kNumLeaves; ++i) {
    HIP_CHECK(hipMalloc(&d_leaves[i], sizeof(int)));
  }

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  BuildFanOutGraph(graph, d_root, d_leaves, kNumLeaves);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Remove stale dot file so runtime writes a fresh one on first launch
  std::string dot_path = PrepareDotFile();
  // Ensure the dot file is removed on scope exit, even if a REQUIRE below throws
  DotFileGuard dot_guard(dot_path);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  std::set<int> stream_ids = ParseSegmentStreams(dot_path);

  // All nodes must be on the same stream — collapse fired
  INFO("Dot file: " << dot_path);
  INFO("Segment stream values found: " << stream_ids.size());
  REQUIRE(stream_ids.size() == 1);

  // Verify functional correctness
  int h_root = 0;
  HIP_CHECK(hipMemcpy(&h_root, d_root, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(h_root == 1);
  for (int i = 0; i < kNumLeaves; ++i) {
    int h_leaf = 0;
    HIP_CHECK(hipMemcpy(&h_leaf, d_leaves[i], sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(h_leaf == i + 10);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_root));
  for (int i = 0; i < kNumLeaves; ++i) HIP_CHECK(hipFree(d_leaves[i]));
  // dot_guard removes the dot file on scope exit (even if a REQUIRE above failed)
}

/**
 * Test: with mode=2 (DFS, no collapse), the same graph uses multiple streams.
 */
TEST_CASE("Unit_hipGraph_CollapseSmallKernels_MultiStream_Mode2", "[graph][collapse][level_2]") {
  // Requires: DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=2 DEBUG_HIP_GRAPH_MIN_OVERLAP=2
  //           DEBUG_HIP_GRAPH_DOT_PRINT=2
  // Flags are read once at HIP init so must be set before process launch;
  // the ctest add_test wiring in CMakeLists.txt provides them.
  // Verifies that mode=2 (DFS, no collapse) uses multiple streams for a fan-out graph.
  constexpr int kNumLeaves = 8;

  int* d_root;
  HIP_CHECK(hipMalloc(&d_root, sizeof(int)));
  std::vector<int*> d_leaves(kNumLeaves);
  for (int i = 0; i < kNumLeaves; ++i) {
    HIP_CHECK(hipMalloc(&d_leaves[i], sizeof(int)));
  }

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  BuildFanOutGraph(graph, d_root, d_leaves, kNumLeaves);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Remove stale dot file so runtime writes a fresh one on first launch
  std::string dot_path = PrepareDotFile();
  // Ensure the dot file is removed on scope exit, even if a REQUIRE below throws
  DotFileGuard dot_guard(dot_path);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::set<int> stream_ids = ParseSegmentStreams(dot_path);

  // Mode=2 should assign multiple streams for this fan-out graph
  INFO("Dot file: " << dot_path);
  INFO("Segment stream values found: " << stream_ids.size());
  REQUIRE(stream_ids.size() > 1);

  // Verify functional correctness still holds
  int h_root = 0;
  HIP_CHECK(hipMemcpy(&h_root, d_root, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(h_root == 1);
  for (int i = 0; i < kNumLeaves; ++i) {
    int h_leaf = 0;
    HIP_CHECK(hipMemcpy(&h_leaf, d_leaves[i], sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(h_leaf == i + 10);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_root));
  for (int i = 0; i < kNumLeaves; ++i) HIP_CHECK(hipFree(d_leaves[i]));
  // dot_guard removes the dot file on scope exit (even if a REQUIRE above failed)
}
