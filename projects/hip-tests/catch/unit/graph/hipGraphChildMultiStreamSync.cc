/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Deterministic correctness tests for multi-stream child graphs on the segmented
 * HIP path. Scheduling flags must be set before HIP init (see CMake add_test).
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
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

__global__ void write_val(int* dst, int v) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *dst = v;
  }
}

__global__ void add_vals(int* out, const int* a, const int* b) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = *a + *b;
  }
}

__global__ void add_three_vals(int* out, const int* a, const int* b, const int* c) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = *a + *b + *c;
  }
}

__global__ void child_branch_a(int* data) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    data[1] = (data[0] == 1) ? 2 : -1;
  }
}

__global__ void child_branch_b(int* data) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    data[2] = (data[0] == 1) ? 3 : -1;
  }
}

__global__ void parent_consumer(int* data) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    data[3] = (data[1] == 2 && data[2] == 3) ? 1 : -1;
  }
}

static void add_kernel_node(hipGraph_t g, hipGraphNode_t* out, hipGraphNode_t* deps, size_t ndeps,
                            int** d_dst, int v) {
  hipKernelNodeParams p{};
  void* args[] = {d_dst, &v};
  p.func = reinterpret_cast<void*>(write_val);
  p.gridDim = dim3(1);
  p.blockDim = dim3(1);
  p.kernelParams = reinterpret_cast<void**>(args);
  HIP_CHECK(hipGraphAddKernelNode(out, g, deps, ndeps, &p));
}

static void add_add_node(hipGraph_t g, hipGraphNode_t* out, hipGraphNode_t* deps, size_t ndeps,
                         int* d_out, int* d_a, int* d_b) {
  hipKernelNodeParams p{};
  void* args[] = {&d_out, &d_a, &d_b};
  p.func = reinterpret_cast<void*>(add_vals);
  p.gridDim = dim3(1);
  p.blockDim = dim3(1);
  p.kernelParams = reinterpret_cast<void**>(args);
  HIP_CHECK(hipGraphAddKernelNode(out, g, deps, ndeps, &p));
}

static std::string PrepareDotFile() {
#if HT_WIN
  int pid = _getpid();
#else
  pid_t pid = getpid();
#endif
  std::string path = "graph_" + std::to_string(pid) + "_dot_print_launch_1";
  std::remove(path.c_str());
  return path;
}

static std::vector<int> ParseParentEntryEvents(const std::string& path) {
  std::ifstream dot(path);
  std::string line;
  const std::regex parent_entry_re("Parent entry event: ([0-9]+)");
  std::vector<int> events;
  while (std::getline(dot, line)) {
    for (std::sregex_iterator it(line.begin(), line.end(), parent_entry_re), end; it != end; ++it) {
      events.push_back(std::stoi((*it)[1].str()));
    }
  }
  return events;
}

static std::set<int> ParseTopLevelSegmentStreams(const std::string& path) {
  std::ifstream dot(path);
  std::string line;
  const std::regex graph_label_re("label=\"graph_[0-9]+\"");
  const std::regex stream_re("Stream: ([0-9]+)");
  std::set<int> streams;
  int graph_count = 0;
  while (std::getline(dot, line)) {
    if (std::regex_search(line, graph_label_re) && ++graph_count == 2) {
      break;
    }
    if (graph_count != 1) {
      continue;
    }
    for (std::sregex_iterator it(line.begin(), line.end(), stream_re), end; it != end; ++it) {
      streams.insert(std::stoi((*it)[1].str()));
    }
  }
  return streams;
}

struct DotFileGuard {
  std::string path;
  explicit DotFileGuard(const std::string& p) : path(p) {}
  ~DotFileGuard() { std::remove(path.c_str()); }
};

#if HT_AMD  // relies on DEBUG_HIP_GRAPH_* flags that are hipamd-only
TEST_CASE("Unit_hipGraphChildMultiStream_ParentChildBoundarySync",
          "[graph][child][multistream][level_2]") {
  int* data = nullptr;
  int* update_scratch = nullptr;
  int* initial_update_source = nullptr;
  int* dummy_result = nullptr;
  HIP_CHECK(hipMalloc(&data, 4 * sizeof(int)));
  HIP_CHECK(hipMalloc(&update_scratch, sizeof(int)));
  HIP_CHECK(hipMalloc(&initial_update_source, sizeof(int)));
  HIP_CHECK(hipMalloc(&dummy_result, sizeof(int)));

  hipGraph_t child = nullptr;
  hipGraph_t updated_child = nullptr;
  hipGraph_t parent = nullptr;
  HIP_CHECK(hipGraphCreate(&child, 0));
  HIP_CHECK(hipGraphCreate(&updated_child, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));

  hipGraphNode_t producer = nullptr;
  add_kernel_node(parent, &producer, nullptr, 0, &data, 1);

  int initial_host_value = 11;
  int updated_host_value = 22;
  HIP_CHECK(hipMemcpy(initial_update_source, &initial_host_value, sizeof(int),
                      hipMemcpyHostToDevice));
  auto addChildRoots = [&](hipGraph_t graph, const void* update_source, hipMemcpyKind update_kind) {
    void* branch_args[] = {&data};
    hipKernelNodeParams branch_a_params{};
    branch_a_params.func = reinterpret_cast<void*>(child_branch_a);
    branch_a_params.gridDim = dim3(1);
    branch_a_params.blockDim = dim3(1);
    branch_a_params.kernelParams = branch_args;
    hipKernelNodeParams branch_b_params = branch_a_params;
    branch_b_params.func = reinterpret_cast<void*>(child_branch_b);

    hipGraphNode_t branch_a = nullptr;
    hipGraphNode_t branch_b = nullptr;
    hipGraphNode_t update_memcpy = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&branch_a, graph, nullptr, 0, &branch_a_params));
    HIP_CHECK(hipGraphAddKernelNode(&branch_b, graph, nullptr, 0, &branch_b_params));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&update_memcpy, graph, nullptr, 0, update_scratch,
                                      update_source, sizeof(int), update_kind));
  };
  addChildRoots(child, initial_update_source, hipMemcpyDeviceToDevice);
  addChildRoots(updated_child, &updated_host_value, hipMemcpyHostToDevice);

  hipGraphNode_t child_node = nullptr;
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, &producer, 1, child));

  // Put one child successor on another stream. The child segment therefore
  // receives a regular completion slot before its entry-event slot, ensuring
  // the parent entry slot exercised by the update path is nonzero.
  hipGraphNode_t dummy_consumer = nullptr;
  add_kernel_node(parent, &dummy_consumer, &child_node, 1, &dummy_result, 7);

  void* consumer_args[] = {&data};
  hipKernelNodeParams consumer_params{};
  consumer_params.func = reinterpret_cast<void*>(parent_consumer);
  consumer_params.gridDim = dim3(1);
  consumer_params.blockDim = dim3(1);
  consumer_params.kernelParams = consumer_args;

  hipGraphNode_t consumer = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&consumer, parent, &child_node, 1, &consumer_params));

  const std::string dot_path = PrepareDotFile();
  DotFileGuard dot_guard(dot_path);

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, parent, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  // Updating an uncaptured H2D node forces a full child packet recapture before
  // the first launch. Recursive enqueue must bind the recreated waits to the
  // immediate parent's current nonzero entry-event slot.
  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(exec, child_node, updated_child));

  for (int iteration = 0; iteration < 2; ++iteration) {
    // Unsynchronized work queued before hipGraphLaunch must be visible to every
    // graph queue before the parent producer and child roots execute.
    HIP_CHECK(hipMemsetAsync(data, 0, 4 * sizeof(int), stream));
    HIP_CHECK(hipMemsetAsync(update_scratch, 0, sizeof(int), stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    if (iteration == 0) {
      const std::vector<int> entry_events = ParseParentEntryEvents(dot_path);
      REQUIRE(entry_events.size() == 2);
      REQUIRE(entry_events[0] > 0);
      REQUIRE(entry_events[1] == entry_events[0]);
    }

    int result[4] = {};
    int scratch = 0;
    HIP_CHECK(hipMemcpyAsync(result, data, sizeof(result), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(&scratch, update_scratch, sizeof(scratch), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    INFO("iteration " << iteration);
    REQUIRE(result[0] == 1);  // parent producer completed
    REQUIRE(result[1] == 2);  // child root A observed parent producer
    REQUIRE(result[2] == 3);  // off-stream child root B observed parent producer
    REQUIRE(result[3] == 1);  // parent consumer observed both child leaves
    REQUIRE(scratch == updated_host_value);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(parent));
  HIP_CHECK(hipGraphDestroy(child));
  HIP_CHECK(hipGraphDestroy(updated_child));
  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(update_scratch));
  HIP_CHECK(hipFree(initial_update_source));
  HIP_CHECK(hipFree(dummy_result));
}

TEST_CASE("Unit_hipGraphChildMultiStream_SiblingChildren", "[graph][child][multistream][level_2]") {
  int *d_a, *d_b, *d_c, *d_sum;
  HIP_CHECK(hipMalloc(&d_a, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_b, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_c, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));

  hipGraph_t child1, child2, child3;
  HIP_CHECK(hipGraphCreate(&child1, 0));
  HIP_CHECK(hipGraphCreate(&child2, 0));
  HIP_CHECK(hipGraphCreate(&child3, 0));

  hipGraphNode_t c1, c2, c3;
  add_kernel_node(child1, &c1, nullptr, 0, &d_a, 10);
  add_kernel_node(child2, &c2, nullptr, 0, &d_b, 20);
  add_kernel_node(child3, &c3, nullptr, 0, &d_c, 30);

  hipGraph_t parent;
  HIP_CHECK(hipGraphCreate(&parent, 0));
  hipGraphNode_t p_child1, p_child2, p_child3;
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child1, parent, nullptr, 0, child1));
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child2, parent, nullptr, 0, child2));
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child3, parent, nullptr, 0, child3));

  hipGraphNode_t deps[3] = {p_child1, p_child2, p_child3};
  void* sum_args[] = {&d_sum, &d_a, &d_b, &d_c};
  hipKernelNodeParams sum_params{};
  sum_params.func = reinterpret_cast<void*>(add_three_vals);
  sum_params.gridDim = dim3(1);
  sum_params.blockDim = dim3(1);
  sum_params.kernelParams = sum_args;
  hipGraphNode_t p_sum;
  HIP_CHECK(hipGraphAddKernelNode(&p_sum, parent, deps, 3, &sum_params));

  const std::string dot_path = PrepareDotFile();
  DotFileGuard dot_guard(dot_path);

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, parent, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int sum = 0;
  HIP_CHECK(hipMemcpy(&sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(sum == 60);
  const std::set<int> expected_streams{0, 1, 2};
  REQUIRE(ParseTopLevelSegmentStreams(dot_path) == expected_streams);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(parent));
  HIP_CHECK(hipGraphDestroy(child1));
  HIP_CHECK(hipGraphDestroy(child2));
  HIP_CHECK(hipGraphDestroy(child3));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  HIP_CHECK(hipFree(d_sum));
}

TEST_CASE("Unit_hipGraphChildMultiStream_NestedChild", "[graph][child][multistream][level_2]") {
  hipGraph_t inner, outer, parent;
  HIP_CHECK(hipGraphCreate(&inner, 0));
  HIP_CHECK(hipGraphCreate(&outer, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));

  int *d_x, *d_y, *d_z, *d_zero, *d_final;
  HIP_CHECK(hipMalloc(&d_x, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_y, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_z, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_zero, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_final, sizeof(int)));
  HIP_CHECK(hipMemset(d_zero, 0, sizeof(int)));

  hipGraphNode_t ix, iy;
  add_kernel_node(inner, &ix, nullptr, 0, &d_x, 5);
  add_kernel_node(inner, &iy, nullptr, 0, &d_y, 7);

  hipGraphNode_t o_child, oz;
  HIP_CHECK(hipGraphAddChildGraphNode(&o_child, outer, nullptr, 0, inner));
  add_add_node(outer, &oz, &o_child, 1, d_z, d_x, d_y);

  hipGraphNode_t p_child, p_consumer;
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child, parent, nullptr, 0, outer));
  add_add_node(parent, &p_consumer, &p_child, 1, d_final, d_z, d_zero);

  const std::string dot_path = PrepareDotFile();
  DotFileGuard dot_guard(dot_path);

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, parent, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int final = 0;
  HIP_CHECK(hipMemcpy(&final, d_final, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(final == 12);
  REQUIRE(ParseParentEntryEvents(dot_path).size() == 1);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(parent));
  HIP_CHECK(hipGraphDestroy(outer));
  HIP_CHECK(hipGraphDestroy(inner));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_z));
  HIP_CHECK(hipFree(d_zero));
  HIP_CHECK(hipFree(d_final));
}
#endif
