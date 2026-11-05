/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphFastLaunch hipGraphFastLaunch
 * @{
 * @ingroup GraphTest
 * Checks the results of multi-stream graph launches on the paths that DEBUG_CLR_GRAPH_FAST_LAUNCH
 * changes: the end marker's gate (also with several threads launching at once), the fork waits of
 * the side streams (folded into the first dispatch where the device uses the ext dispatch packet),
 * an empty node carrying its segment's cross-stream waits, and the collapse of a graph with a
 * cross-stream edge into a node that isn't AQL-captured (the default collapse rule, C1). Each test
 * runs with DEBUG_CLR_GRAPH_FAST_LAUNCH=0 and =1 (ctest entries in CMakeLists.txt; the flags are
 * read once, at HIP init). The kernels spin for different times so that a missing wait lets a fast
 * kernel overtake a slow one, and the device and the host check counts that only add up if every
 * wait held. The segment layout is read from the dot file DEBUG_HIP_GRAPH_DOT_PRINT=2 writes at the
 * first launch.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#if HT_WIN
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

struct Segment {
  int stream;
  int level;
};

// Segments of the graph dumped by DEBUG_HIP_GRAPH_DOT_PRINT=2 ("Segment <id>\nStream: <s>\n
// Level: <l>" in each cluster label)
std::map<int, Segment> ParseSegments(const std::string& path) {
  std::ifstream f(path);
  std::map<int, Segment> segments;
  std::string line;
  std::regex re(R"(Segment (\d+)\\nStream: (\d+)\\nLevel: (\d+))");
  while (std::getline(f, line)) {
    std::smatch m;
    if (std::regex_search(line, m, re)) {
      segments[std::stoi(m[1].str())] = {std::stoi(m[2].str()), std::stoi(m[3].str())};
    }
  }
  return segments;
}

std::set<int> Streams(const std::map<int, Segment>& segments) {
  std::set<int> streams;
  for (const auto& [id, seg] : segments) streams.insert(seg.stream);
  return streams;
}

// The dot file of the next first launch; a stale copy is removed first, and the guard removes
// this one on scope exit
struct DotFile {
  std::string path;
  DotFile() {
#if HT_WIN
    int pid = _getpid();
#else
    pid_t pid = getpid();
#endif
    path = "graph_" + std::to_string(pid) + "_dot_print_launch_1";
    std::remove(path.c_str());
  }
  ~DotFile() { std::remove(path.c_str()); }
};

// The head of a chain checks that every chain finished at least as many launches as its own
// (the previous launch's tails are all done); the tail spins, then counts the launch
__global__ void ChainHead(volatile int* count, int chain, int num_chains, int* errors) {
  if (threadIdx.x == 0) {
    for (int j = 0; j < num_chains; ++j) {
      if (count[j] < count[chain]) atomicAdd(errors, 1);
    }
  }
}

__global__ void ChainTail(volatile int* count, int chain, long long spin) {
  if (threadIdx.x == 0) {
    const long long t0 = clock64();
    while (clock64() - t0 < spin) {
    }
    count[chain] = count[chain] + 1;
  }
}

// A root spins, then counts its launch; the checker after the join requires every root to be one
// launch ahead of it, then counts its own launch
__global__ void JoinRoot(volatile int* count, int root, long long spin) {
  if (threadIdx.x == 0) {
    const long long t0 = clock64();
    while (clock64() - t0 < spin) {
    }
    count[root] = count[root] + 1;
  }
}

__global__ void JoinCheck(volatile int* count, int num_roots, int* errors) {
  if (threadIdx.x == 0) {
    for (int j = 0; j < num_roots; ++j) {
      if (count[j] != count[num_roots] + 1) atomicAdd(errors, 1);
    }
    count[num_roots] = count[num_roots] + 1;
  }
}

__global__ void FillValue(int* data, int value, long long spin) {
  if (threadIdx.x == 0) {
    const long long t0 = clock64();
    while (clock64() - t0 < spin) {
    }
  }
  data[threadIdx.x] = value + threadIdx.x;
}

// Spins, then writes value to data[index]
__global__ void SetValue(int* data, int index, int value, long long spin) {
  if (threadIdx.x == 0) {
    const long long t0 = clock64();
    while (clock64() - t0 < spin) {
    }
    data[index] = value;
  }
}

__global__ void SumValues(const int* data, int count, int* sum) {
  if (threadIdx.x == 0) {
    int s = 0;
    for (int i = 0; i < count; ++i) s += data[i];
    *sum = s;
  }
}

struct CopyCheck {
  const int* out;
  int elems;
  int expected;
  int failures;
};

struct HostCheck {
  volatile int* count;
  int num_chains;
  int expected;
  int failures;
};

void CheckCounts(void* data) {
  auto* check = static_cast<HostCheck*>(data);
  for (int j = 0; j < check->num_chains; ++j) {
    if (check->count[j] != check->expected) ++check->failures;
  }
}

hipGraphNode_t AddKernel(hipGraph_t graph, const hipGraphNode_t* deps, size_t num_deps, void* func,
                         void** args) {
  hipKernelNodeParams params{};
  params.func = func;
  params.gridDim = dim3(1);
  params.blockDim = dim3(64);
  params.kernelParams = args;
  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps, num_deps, &params));
  return node;
}

// num_chains independent chains head -> tail of different lengths, with two executable graphs of
// it that launches alternate between, and the counts the kernels update
struct Chains {
  int num_chains;
  int* count;
  int* errors;
  hipGraph_t graph;
  hipGraphExec_t exec[2];
  std::vector<int> chain;
  std::vector<long long> spin;
  std::vector<std::vector<void*>> head_args, tail_args;

  explicit Chains(int n) : num_chains(n), chain(n), spin(n), head_args(n), tail_args(n) {
    HIP_CHECK(hipHostMalloc(&count, num_chains * sizeof(int), hipHostMallocCoherent));
    for (int j = 0; j < num_chains; ++j) count[j] = 0;
    HIP_CHECK(hipMalloc(&errors, sizeof(int)));
    HIP_CHECK(hipMemset(errors, 0, sizeof(int)));
    HIP_CHECK(hipGraphCreate(&graph, 0));
    for (int j = 0; j < num_chains; ++j) {
      chain[j] = j;
      spin[j] = 20000 + 40000LL * (j % 3);
      head_args[j] = {&count, &chain[j], &num_chains, &errors};
      tail_args[j] = {&count, &chain[j], &spin[j]};
      hipGraphNode_t head =
          AddKernel(graph, nullptr, 0, reinterpret_cast<void*>(ChainHead), head_args[j].data());
      AddKernel(graph, &head, 1, reinterpret_cast<void*>(ChainTail), tail_args[j].data());
    }
    HIP_CHECK(hipGraphInstantiate(&exec[0], graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphInstantiate(&exec[1], graph, nullptr, nullptr, 0));
  }
  void Destroy() {
    HIP_CHECK(hipGraphExecDestroy(exec[0]));
    HIP_CHECK(hipGraphExecDestroy(exec[1]));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipFree(errors));
    HIP_CHECK(hipHostFree(count));
  }
  int DeviceErrors() const {
    int device_errors = 0;
    HIP_CHECK(hipMemcpy(&device_errors, errors, sizeof(int), hipMemcpyDeviceToHost));
    return device_errors;
  }
};

// Chains launched back to back on one stream. A host function after every third launch checks
// the counts on the host, and the stream is synchronized every 100 launches; the heads check every
// launch on the device. Returns the segments of the graph's first launch.
std::map<int, Segment> RunChains(int num_chains, int launches) {
  Chains chains(num_chains);
  DotFile dot;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  std::vector<HostCheck> checks(launches / 3 + 1);
  for (int k = 1; k <= launches; ++k) {
    HIP_CHECK(hipGraphLaunch(chains.exec[k & 1], stream));
    if (k == 1) HIP_CHECK(hipStreamSynchronize(stream));
    if (k % 3 == 0) {
      checks[k / 3] = {chains.count, num_chains, k, 0};
      HIP_CHECK(hipLaunchHostFunc(stream, CheckCounts, &checks[k / 3]));
    }
    if (k % 100 == 0) HIP_CHECK(hipStreamSynchronize(stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  std::map<int, Segment> segments = ParseSegments(dot.path);

  int host_failures = 0;
  for (const auto& check : checks) host_failures += check.failures;
  const int device_errors = chains.DeviceErrors();
  INFO("device check errors " << device_errors << ", host check failures " << host_failures);
  REQUIRE(device_errors == 0);
  REQUIRE(host_failures == 0);
  for (int j = 0; j < num_chains; ++j) REQUIRE(chains.count[j] == launches);

  HIP_CHECK(hipStreamDestroy(stream));
  chains.Destroy();
  return segments;
}

}  // namespace

#if HT_AMD  // relies on DEBUG_HIP_GRAPH_* flags that are hipamd-only
/**
 * Test Description
 * ------------------------
 *  - Three chains launched 2000 times back to back on one stream, alternating two executable
 *    graphs: each launch's side streams wait on the previous launch's end marker (its gate with
 *    DEBUG_CLR_GRAPH_FAST_LAUNCH=1), whose commands the launch holds until OnLaunchComplete.
 *  - Every head checks on the device that all chains finished the previous launch; host
 *    functions and synchronizations check the counts on the host.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphFastLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18; DEBUG_HIP_GRAPH_MIN_OVERLAP=0 DEBUG_HIP_GRAPH_DOT_PRINT=2 (ctest)
 */
TEST_CASE("Unit_hipGraph_FastLaunch_BackToBackChains", "[graph][fast_launch][level_2]") {
  std::map<int, Segment> segments = RunChains(3, 2000);
  INFO("segments in the dot file: " << segments.size());
  REQUIRE(Streams(segments).size() > 1);
}

/**
 * Test Description
 * ------------------------
 *  - Eight chains: more level-0 segments than graph streams (DEBUG_HIP_FORCE_GRAPH_QUEUES, 4),
 *    so side streams carry two level-0 segments that wait on the same command: the second fork
 *    wait meets the pending one, which the side stream's first batch then carries or flushes.
 *  - Launched 1000 times back to back, results checked as in BackToBackChains.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphFastLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18; DEBUG_HIP_GRAPH_MIN_OVERLAP=0 DEBUG_HIP_GRAPH_DOT_PRINT=2 (ctest)
 */
TEST_CASE("Unit_hipGraph_FastLaunch_SharedSideStream", "[graph][fast_launch][level_2]") {
  std::map<int, Segment> segments = RunChains(8, 1000);
  std::map<int, int> level0_per_stream;
  for (const auto& [id, seg] : segments) {
    if (seg.level == 0) ++level0_per_stream[seg.stream];
  }
  bool shared_side_stream = false;
  for (const auto& [stream, n] : level0_per_stream) {
    if (stream != 0 && n >= 2) shared_side_stream = true;
  }
  INFO("segments in the dot file: " << segments.size());
  REQUIRE(shared_side_stream);
}

/**
 * Test Description
 * ------------------------
 *  - Four threads, each launching its own three chains (two executable graphs) 1000 times back to
 *    back on its own stream: the end markers of the four streams take their gates from the
 *    device's one pool at the same time, and each gate must belong to one launch only, or the
 *    first end marker to complete releases another launch's waiters early.
 *  - The first launch of each executable graph is made beforehand, one at a time (it writes the
 *    dot file). Results checked per thread as in BackToBackChains.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphFastLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18; DEBUG_HIP_GRAPH_MIN_OVERLAP=0 DEBUG_HIP_GRAPH_DOT_PRINT=2 (ctest)
 */
TEST_CASE("Unit_hipGraph_FastLaunch_ConcurrentLaunchers", "[graph][fast_launch][level_2]") {
  constexpr int kThreads = 4;
  constexpr int kChains = 3;
  constexpr int kLaunches = 1000;
  DotFile dot;
  std::vector<std::unique_ptr<Chains>> chains;
  std::vector<hipStream_t> streams(kThreads);
  std::vector<std::vector<HostCheck>> checks(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    chains.push_back(std::make_unique<Chains>(kChains));
    HIP_CHECK(hipStreamCreateWithFlags(&streams[t], hipStreamNonBlocking));
    checks[t].resize(kLaunches / 3 + 1);
  }
  // The first launch of each executable graph writes the dot file: one at a time, before the
  // threads start
  constexpr int kFirstLaunches = 2;
  std::map<int, Segment> segments;
  for (int t = 0; t < kThreads; ++t) {
    for (int e = 0; e < kFirstLaunches; ++e) {
      HIP_CHECK(hipGraphLaunch(chains[t]->exec[e], streams[t]));
      HIP_CHECK(hipStreamSynchronize(streams[t]));
      if (t == 0 && e == 0) segments = ParseSegments(dot.path);
    }
  }
  auto launcher = [&](int t) {
    for (int k = kFirstLaunches + 1; k <= kLaunches; ++k) {
      HIP_CHECK_THREAD(hipGraphLaunch(chains[t]->exec[k & 1], streams[t]));
      if (k % 3 == 0) {
        checks[t][k / 3] = {chains[t]->count, kChains, k, 0};
        HIP_CHECK_THREAD(hipLaunchHostFunc(streams[t], CheckCounts, &checks[t][k / 3]));
      }
      if (k % 100 == 0) HIP_CHECK_THREAD(hipStreamSynchronize(streams[t]));
    }
    HIP_CHECK_THREAD(hipStreamSynchronize(streams[t]));
  };
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(launcher, t);
  for (auto& thread : threads) thread.join();
  HIP_CHECK_THREAD_FINALIZE();

  for (int t = 0; t < kThreads; ++t) {
    int host_failures = 0;
    for (const auto& check : checks[t]) host_failures += check.failures;
    const int device_errors = chains[t]->DeviceErrors();
    int short_counts = 0;
    for (int j = 0; j < kChains; ++j) short_counts += (chains[t]->count[j] != kLaunches);
    INFO("thread " << t << ": device check errors " << device_errors << ", host check failures "
                   << host_failures << ", chains short of the launches " << short_counts);
    CHECK(device_errors == 0);
    CHECK(host_failures == 0);
    CHECK(short_counts == 0);
  }
  INFO("segments in the dot file: " << segments.size());
  REQUIRE(Streams(segments).size() > 1);

  for (int t = 0; t < kThreads; ++t) {
    HIP_CHECK(hipStreamDestroy(streams[t]));
    chains[t]->Destroy();
  }
}

/**
 * Test Description
 * ------------------------
 *  - An empty node joins 2, 3, 4, 5 and 6 roots on other streams (up to five waits ride in the
 *    empty node's own barrier-AND with DEBUG_CLR_GRAPH_FAST_LAUNCH=1; six keep a barrier before
 *    it), followed by a checker kernel that requires every root of the same launch to be done.
 *  - Launched back to back; every 10 launches the checker is disabled or enabled again with
 *    hipGraphNodeSetEnabled, which moves the segment's completion signal onto the empty node
 *    while the checker is off. The counts on the device and on the host must match the launches.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphFastLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18; DEBUG_HIP_GRAPH_MIN_OVERLAP=0 DEBUG_HIP_GRAPH_DOT_PRINT=2 (ctest)
 */
TEST_CASE("Unit_hipGraph_FastLaunch_EmptyNodeJoin", "[graph][fast_launch][level_2]") {
  const int num_roots = GENERATE(2, 3, 4, 5, 6);
  constexpr int kLaunches = 600;
  int* count;
  int* errors;
  HIP_CHECK(hipMalloc(&count, 8 * sizeof(int)));
  HIP_CHECK(hipMemset(count, 0, 8 * sizeof(int)));
  HIP_CHECK(hipMalloc(&errors, sizeof(int)));
  HIP_CHECK(hipMemset(errors, 0, sizeof(int)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<int> root(num_roots);
  std::vector<long long> spin(num_roots);
  std::vector<std::vector<void*>> root_args(num_roots);
  std::vector<hipGraphNode_t> roots(num_roots);
  for (int j = 0; j < num_roots; ++j) {
    root[j] = j;
    spin[j] = 20000 + 15000LL * j;
    root_args[j] = {&count, &root[j], &spin[j]};
    roots[j] = AddKernel(graph, nullptr, 0, reinterpret_cast<void*>(JoinRoot), root_args[j].data());
  }
  hipGraphNode_t join;
  HIP_CHECK(hipGraphAddEmptyNode(&join, graph, roots.data(), roots.size()));
  int n = num_roots;
  void* check_args[] = {&count, &n, &errors};
  hipGraphNode_t check = AddKernel(graph, &join, 1, reinterpret_cast<void*>(JoinCheck), check_args);
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  DotFile dot;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int checker_launches = 0;
  bool checker_on = true;
  for (int k = 1; k <= kLaunches; ++k) {
    if (k % 10 == 1 && k > 1) {
      // Each launch copies the packets, so a change applies from the next launch on
      checker_on = !checker_on;
      HIP_CHECK(hipGraphNodeSetEnabled(exec, check, checker_on ? 1 : 0));
      if (!checker_on) {
        // The checker requires the roots one launch ahead of it: resynchronize its count first
        HIP_CHECK(hipStreamSynchronize(stream));
      }
    }
    HIP_CHECK(hipGraphLaunch(exec, stream));
    if (checker_on) ++checker_launches;
    if (!checker_on) {
      // With the checker off, the end of the launch must still follow every root
      HIP_CHECK(hipStreamSynchronize(stream));
      std::vector<int> h(num_roots);
      HIP_CHECK(hipMemcpy(h.data(), count, num_roots * sizeof(int), hipMemcpyDeviceToHost));
      for (int j = 0; j < num_roots; ++j) REQUIRE(h[j] == k);
      // Bring the checker's count to the roots' minus one for its next launch
      int ahead = k;
      HIP_CHECK(hipMemcpy(count + num_roots, &ahead, sizeof(int), hipMemcpyHostToDevice));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  std::map<int, Segment> segments = ParseSegments(dot.path);

  int device_errors = 0;
  std::vector<int> h(num_roots + 1);
  HIP_CHECK(hipMemcpy(&device_errors, errors, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h.data(), count, (num_roots + 1) * sizeof(int), hipMemcpyDeviceToHost));
  INFO("roots " << num_roots << ", checker launches " << checker_launches
                << ", device check errors " << device_errors);
  REQUIRE(device_errors == 0);
  for (int j = 0; j < num_roots; ++j) REQUIRE(h[j] == kLaunches);
  REQUIRE(Streams(segments).size() > 1);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(errors));
  HIP_CHECK(hipFree(count));
}

/**
 * Test Description
 * ------------------------
 *  - Two independent kernels on separate chains feed a device-to-host memcpy node and a host
 *    node: cross-stream edges into nodes that aren't AQL-captured. With the default collapse
 *    rule (C1) the graph collapses onto the launch stream (one stream in the dot file); the
 *    copied data and the host node's view of it are checked on every launch.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphFastLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18; DEBUG_HIP_GRAPH_DOT_PRINT=2 (ctest), DEBUG_HIP_GRAPH_COLLAPSE_RULE
 *    and DEBUG_HIP_GRAPH_MIN_OVERLAP unset
 */
TEST_CASE("Unit_hipGraph_FastLaunch_UncapturedEdgeCollapse", "[graph][fast_launch][level_2]") {
  constexpr int kElems = 64;
  constexpr int kLaunches = 200;
  int* d_a;
  int* d_b;
  int* h_out;
  HIP_CHECK(hipMalloc(&d_a, kElems * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_b, kElems * sizeof(int)));
  HIP_CHECK(hipHostMalloc(&h_out, 2 * kElems * sizeof(int)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int value_a = 0, value_b = 0;
  long long spin_a = 60000, spin_b = 5000;
  void* args_a[] = {&d_a, &value_a, &spin_a};
  void* args_b[] = {&d_b, &value_b, &spin_b};
  hipGraphNode_t fill_a = AddKernel(graph, nullptr, 0, reinterpret_cast<void*>(FillValue), args_a);
  hipGraphNode_t fill_b = AddKernel(graph, nullptr, 0, reinterpret_cast<void*>(FillValue), args_b);
  hipGraphNode_t copy_a, copy_b;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&copy_a, graph, &fill_a, 1, h_out, d_a, kElems * sizeof(int),
                                    hipMemcpyDeviceToHost));
  hipGraphNode_t both[] = {fill_a, fill_b};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&copy_b, graph, both, 2, h_out + kElems, d_b,
                                    kElems * sizeof(int), hipMemcpyDeviceToHost));
  // The host node sees both copies of the same launch
  CopyCheck host_check{h_out, kElems, 0, 0};
  hipHostNodeParams host_params{};
  host_params.fn = [](void* data) {
    auto* check = static_cast<CopyCheck*>(data);
    for (int i = 0; i < check->elems; ++i) {
      if (check->out[i] != check->expected + i) ++check->failures;
      if (check->out[check->elems + i] != check->expected + 1000 + i) ++check->failures;
    }
  };
  host_params.userData = &host_check;
  hipGraphNode_t copies[] = {copy_a, copy_b}, host;
  HIP_CHECK(hipGraphAddHostNode(&host, graph, copies, 2, &host_params));

  DotFile dot;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  int copy_failures = 0;
  for (int k = 1; k <= kLaunches; ++k) {
    value_a = 10 * k;
    value_b = 10 * k + 1000;
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(FillValue);
    params.gridDim = dim3(1);
    params.blockDim = dim3(kElems);
    params.kernelParams = args_a;
    HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, fill_a, &params));
    params.kernelParams = args_b;
    HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, fill_b, &params));
    host_check.expected = 10 * k;
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (int i = 0; i < kElems; ++i) {
      if (h_out[i] != 10 * k + i) ++copy_failures;
      if (h_out[kElems + i] != 10 * k + 1000 + i) ++copy_failures;
    }
  }
  std::map<int, Segment> segments = ParseSegments(dot.path);
  INFO("copy failures " << copy_failures << ", host node failures " << host_check.failures
                        << ", segments in the dot file " << segments.size());
  REQUIRE(copy_failures == 0);
  REQUIRE(host_check.failures == 0);
  REQUIRE(!segments.empty());
  REQUIRE(Streams(segments).size() == 1);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipHostFree(h_out));
}

/**
 * Test Description
 * ------------------------
 *  - A root kernel feeds two branches on separate streams, a chain of two kernels and a child
 *    graph node, joined by a kernel: cross-stream edges into and out of the child. The child
 *    holds two kernels, a kernel then a device-to-host memcpy, or a kernel then a host node.
 *  - A child graph node synchronizes its own nodes, so it doesn't count as a node that isn't
 *    AQL-captured: the graph stays on several streams (dot file). Launched back to back and
 *    synchronized, the joined values and the copy are checked.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphFastLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18; DEBUG_HIP_GRAPH_MIN_OVERLAP=0 DEBUG_HIP_GRAPH_DOT_PRINT=2 (ctest)
 */
TEST_CASE("Unit_hipGraph_FastLaunch_ChildGraphEdges", "[graph][fast_launch][level_2]") {
  enum Child { kKernels, kMemcpy, kHost };
  const Child kind = GENERATE(kKernels, kMemcpy, kHost);
  constexpr int kLaunches = 200;
  int* d_vals;
  int* d_sum;
  int* h_copy;
  HIP_CHECK(hipMalloc(&d_vals, 8 * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));
  HIP_CHECK(hipHostMalloc(&h_copy, 8 * sizeof(int)));
  static int host_calls = 0;
  host_calls = 0;

  // value[i] = 10 + i, written by the kernel of node i after its spin
  int index[6];
  long long spin[6];
  int value[6];
  std::vector<std::vector<void*>> args(6);
  auto value_kernel = [&](hipGraph_t graph, const hipGraphNode_t* deps, size_t num_deps, int i) {
    index[i] = i;
    value[i] = 10 + i;
    spin[i] = 20000;
    args[i] = {&d_vals, &index[i], &value[i], &spin[i]};
    return AddKernel(graph, deps, num_deps, reinterpret_cast<void*>(SetValue), args[i].data());
  };
  hipGraph_t child;
  HIP_CHECK(hipGraphCreate(&child, 0));
  hipGraphNode_t c0 = value_kernel(child, nullptr, 0, 4);
  int expected = 10 + 11 + 12 + 14;
  if (kind == kKernels) {
    value_kernel(child, &c0, 1, 5);
    expected += 15;
  } else if (kind == kMemcpy) {
    hipGraphNode_t copy;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&copy, child, &c0, 1, h_copy, d_vals, 8 * sizeof(int),
                                      hipMemcpyDeviceToHost));
  } else {
    hipHostNodeParams host_params{};
    host_params.fn = [](void*) { ++host_calls; };
    hipGraphNode_t host;
    HIP_CHECK(hipGraphAddHostNode(&host, child, &c0, 1, &host_params));
  }
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t k0 = value_kernel(graph, nullptr, 0, 0);
  hipGraphNode_t k1 = value_kernel(graph, &k0, 1, 1);
  hipGraphNode_t k2 = value_kernel(graph, &k1, 1, 2);
  hipGraphNode_t child_node;
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, graph, &k0, 1, child));
  hipGraphNode_t joined[] = {k2, child_node};
  int num_vals = 6;
  void* sum_args[] = {&d_vals, &num_vals, &d_sum};
  AddKernel(graph, joined, 2, reinterpret_cast<void*>(SumValues), sum_args);
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  DotFile dot;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int failures = 0;
  for (int k = 1; k <= kLaunches; ++k) {
    HIP_CHECK(hipMemsetAsync(d_vals, 0, 8 * sizeof(int), stream));
    HIP_CHECK(hipMemsetAsync(d_sum, 0, sizeof(int), stream));
    std::fill(h_copy, h_copy + 8, 0);
    HIP_CHECK(hipGraphLaunch(exec, stream));
    if (k % 10 == 0) {
      // Back to back: nine launches without a check
      for (int b = 0; b < 9; ++b) HIP_CHECK(hipGraphLaunch(exec, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    int sum = 0;
    HIP_CHECK(hipMemcpy(&sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));
    if (sum != expected) ++failures;
    if ((kind == kMemcpy) && (h_copy[4] != 14)) ++failures;
  }
  std::map<int, Segment> segments = ParseSegments(dot.path);
  INFO("child " << kind << ", failures " << failures << ", host node calls " << host_calls
                << ", segments in the dot file " << segments.size());
  REQUIRE(failures == 0);
  if (kind == kHost) REQUIRE(host_calls == kLaunches + 9 * (kLaunches / 10));
  REQUIRE(Streams(segments).size() > 1);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(child));
  HIP_CHECK(hipFree(d_vals));
  HIP_CHECK(hipFree(d_sum));
  HIP_CHECK(hipHostFree(h_copy));
}
#endif  // HT_AMD

/**
 * End doxygen group GraphTest.
 * @}
 */
