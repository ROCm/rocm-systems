/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Deterministic correctness tests for multi-stream child graphs on the segmented
 * HIP path. Scheduling flags must be set before HIP init (see CMake add_test).
 *
 * Producers and selected child leaves block on host-released gates. A consumer
 * that is not ordered after a gated kernel runs while that kernel is still held,
 * so a missing parent->child or child->parent barrier fails independently of
 * kernel duration or of which queue the CP services first.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <chrono>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#if HT_WIN
#include <process.h>
#else
#include <unistd.h>
#endif

#if HT_AMD  // relies on DEBUG_HIP_GRAPH_* flags that are hipamd-only
// Host-coherent gate state shared with gated kernels. A gated kernel gives up
// after timeout_ticks of wall clock and sets timed_out instead of hanging.
struct GateState {
  static constexpr int kNumGates = 2;
  int gate[kNumGates];
  int timed_out;
  long long timeout_ticks;
};

__device__ void wait_gate(GateState* gs, int gate) {
  const long long start = wall_clock64();
  while (__hip_atomic_load(&gs->gate[gate], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
    if (wall_clock64() - start > gs->timeout_ticks) {
      __hip_atomic_store(&gs->timed_out, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      return;
    }
    __builtin_amdgcn_s_sleep(8);
  }
}

__global__ void write_val(int* dst, int v) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *dst = v;
  }
}

__global__ void gated_write(GateState* gs, int gate, int* dst, int v) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    wait_gate(gs, gate);
    *dst = v;
  }
}

// Reads |src| when the kernel starts, then holds on |gate|. The early read
// catches a missing wait on the producer of |src|; the hold keeps this kernel
// unfinished so a consumer that does not wait for it reads a stale |dst|.
__global__ void gated_check_write(GateState* gs, int gate, int* src, int expected, int* dst,
                                  int v) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int seen = __hip_atomic_load(src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    wait_gate(gs, gate);
    *dst = (seen == expected) ? v : -1;
  }
}

__global__ void gated_zero3(GateState* gs, int gate, int* a, int* b, int* c) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    wait_gate(gs, gate);
    *a = 0;
    *b = 0;
    *c = 0;
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

__global__ void parent_consumer(int* data, int slot) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    data[slot] = (data[1] == 2 && data[2] == 3) ? 1 : -1;
  }
}

class HostGates {
 public:
  static constexpr auto kHold = std::chrono::milliseconds(50);
  static constexpr int kTimeoutSeconds = 10;

  HostGates() {
    HIP_CHECK(hipHostMalloc(&state_, sizeof(GateState), hipHostMallocCoherent));
    int rate_khz = 0;
    HIP_CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
    state_->timeout_ticks = static_cast<long long>(rate_khz) * 1000 * kTimeoutSeconds;
    Reset();
  }
  ~HostGates() { static_cast<void>(hipHostFree(state_)); }

  GateState* get() { return state_; }

  // Only call while no gated kernel is in flight.
  void Reset() {
    for (int& gate : state_->gate) {
      __atomic_store_n(&gate, 0, __ATOMIC_RELAXED);
    }
    __atomic_store_n(&state_->timed_out, 0, __ATOMIC_RELAXED);
  }

  // Holds the gate closed long enough for any unsynchronized consumer to run.
  void Release(int gate) {
    std::this_thread::sleep_for(kHold);
    __atomic_store_n(&state_->gate[gate], 1, __ATOMIC_RELEASE);
  }

  bool TimedOut() const { return __atomic_load_n(&state_->timed_out, __ATOMIC_ACQUIRE) != 0; }

 private:
  GateState* state_ = nullptr;
};

static void add_kernel_node(hipGraph_t g, hipGraphNode_t* out, hipGraphNode_t* deps, size_t ndeps,
                            void* func, void** args) {
  hipKernelNodeParams p{};
  p.func = func;
  p.gridDim = dim3(1);
  p.blockDim = dim3(1);
  p.kernelParams = args;
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

/**
 * producer -> [filler x child_stream] + child_node -> consumer_a, consumer_b
 *
 * Under round-robin the fillers take the lower level-1 streams, so the child
 * node runs on parent stream |child_stream|. Its three roots fan out onto that
 * stream and the next two, and consumer_a/consumer_b take streams 0 and 1, so
 * at least one consumer waits on the child across streams.
 */
TEST_CASE("Unit_hipGraphChildMultiStream_ParentChildBoundarySync",
          "[graph][child][multistream][level_2]") {
  const int child_stream = GENERATE(0, 1, 2);
  INFO("child node on parent stream " << child_stream);
  constexpr int kProducerGate = 0;
  constexpr int kChildLeafGate = 1;

  HostGates gates;
  GateState* gs = gates.get();
  int* data = nullptr;
  int* update_scratch = nullptr;
  int* initial_update_source = nullptr;
  int* filler_result = nullptr;
  HIP_CHECK(hipMalloc(&data, 5 * sizeof(int)));
  HIP_CHECK(hipMalloc(&update_scratch, sizeof(int)));
  HIP_CHECK(hipMalloc(&initial_update_source, sizeof(int)));
  HIP_CHECK(hipMalloc(&filler_result, sizeof(int)));

  hipGraph_t child = nullptr;
  hipGraph_t updated_child = nullptr;
  hipGraph_t parent = nullptr;
  HIP_CHECK(hipGraphCreate(&child, 0));
  HIP_CHECK(hipGraphCreate(&updated_child, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));

  int producer_gate = kProducerGate;
  int one = 1;
  int* data0 = data;
  void* producer_args[] = {&gs, &producer_gate, &data0, &one};
  hipGraphNode_t producer = nullptr;
  add_kernel_node(parent, &producer, nullptr, 0, reinterpret_cast<void*>(gated_write),
                  producer_args);

  int filler_value = 7;
  void* filler_args[] = {&filler_result, &filler_value};
  for (int i = 0; i < child_stream; ++i) {
    hipGraphNode_t filler = nullptr;
    add_kernel_node(parent, &filler, &producer, 1, reinterpret_cast<void*>(write_val),
                    filler_args);
  }

  int initial_host_value = 11;
  int updated_host_value = 22;
  HIP_CHECK(hipMemcpy(initial_update_source, &initial_host_value, sizeof(int),
                      hipMemcpyHostToDevice));

  // branch_a runs on the child node's stream; branch_b and the update memcpy
  // are off-stream roots that must wait on the child entry signal. branch_b is
  // also a gated leaf, so consumers must wait for the whole child.
  int leaf_gate = kChildLeafGate;
  int three = 3;
  int* data2 = data + 2;
  void* branch_a_args[] = {&data};
  void* branch_b_args[] = {&gs, &leaf_gate, &data0, &one, &data2, &three};
  auto addChildRoots = [&](hipGraph_t graph, const void* update_source, hipMemcpyKind update_kind) {
    hipGraphNode_t branch_a = nullptr;
    hipGraphNode_t branch_b = nullptr;
    hipGraphNode_t update_memcpy = nullptr;
    add_kernel_node(graph, &branch_a, nullptr, 0, reinterpret_cast<void*>(child_branch_a),
                    branch_a_args);
    add_kernel_node(graph, &branch_b, nullptr, 0, reinterpret_cast<void*>(gated_check_write),
                    branch_b_args);
    HIP_CHECK(hipGraphAddMemcpyNode1D(&update_memcpy, graph, nullptr, 0, update_scratch,
                                      update_source, sizeof(int), update_kind));
  };
  addChildRoots(child, initial_update_source, hipMemcpyDeviceToDevice);
  addChildRoots(updated_child, &updated_host_value, hipMemcpyHostToDevice);

  hipGraphNode_t child_node = nullptr;
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, &producer, 1, child));

  int slot_a = 3;
  int slot_b = 4;
  void* consumer_a_args[] = {&data, &slot_a};
  void* consumer_b_args[] = {&data, &slot_b};
  hipGraphNode_t consumer_a = nullptr;
  hipGraphNode_t consumer_b = nullptr;
  add_kernel_node(parent, &consumer_a, &child_node, 1, reinterpret_cast<void*>(parent_consumer),
                  consumer_a_args);
  add_kernel_node(parent, &consumer_b, &child_node, 1, reinterpret_cast<void*>(parent_consumer),
                  consumer_b_args);

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, parent, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  // Updating an uncaptured H2D node forces a full child packet recapture before
  // the first launch. Recursive enqueue must bind the recreated child entry
  // waits to the immediate parent's current entry signal.
  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(exec, child_node, updated_child));

  for (int iteration = 0; iteration < 2; ++iteration) {
    INFO("iteration " << iteration);
    gates.Reset();
    HIP_CHECK(hipMemset(data, 0, 5 * sizeof(int)));
    // Unsynchronized work queued before hipGraphLaunch must be visible to every
    // graph queue before the parent producer and child roots execute.
    HIP_CHECK(hipMemsetAsync(update_scratch, 0, sizeof(int), stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    gates.Release(kProducerGate);
    gates.Release(kChildLeafGate);

    int result[5] = {};
    int scratch = 0;
    HIP_CHECK(hipMemcpyAsync(result, data, sizeof(result), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(&scratch, update_scratch, sizeof(scratch), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE_FALSE(gates.TimedOut());
    REQUIRE(result[0] == 1);  // parent producer completed
    REQUIRE(result[1] == 2);  // on-stream child root observed the producer
    REQUIRE(result[2] == 3);  // off-stream child root observed the producer
    REQUIRE(result[3] == 1);  // consumer_a observed both child leaves
    REQUIRE(result[4] == 1);  // consumer_b observed both child leaves
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
  HIP_CHECK(hipFree(filler_result));
}

/**
 * Three sibling child graphs on parent streams 0, 1 and 2, joined by a sum on
 * stream 0. Gated work queued on the launch stream before the graph launch
 * zeroes the child outputs, so children on streams 1 and 2 must wait for it.
 * child2 is gated, so the sum must wait for that child to finish.
 */
TEST_CASE("Unit_hipGraphChildMultiStream_SiblingChildren", "[graph][child][multistream][level_2]") {
  constexpr int kPreLaunchGate = 0;
  constexpr int kChildGate = 1;

  HostGates gates;
  GateState* gs = gates.get();
  int *d_a, *d_b, *d_c, *d_sum;
  HIP_CHECK(hipMalloc(&d_a, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_b, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_c, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));

  hipGraph_t child1, child2, child3;
  HIP_CHECK(hipGraphCreate(&child1, 0));
  HIP_CHECK(hipGraphCreate(&child2, 0));
  HIP_CHECK(hipGraphCreate(&child3, 0));

  int v1 = 10, v2 = 20, v3 = 30;
  int child_gate = kChildGate;
  void* c1_args[] = {&d_a, &v1};
  void* c2_args[] = {&gs, &child_gate, &d_b, &v2};
  void* c3_args[] = {&d_c, &v3};
  hipGraphNode_t c1, c2, c3;
  add_kernel_node(child1, &c1, nullptr, 0, reinterpret_cast<void*>(write_val), c1_args);
  add_kernel_node(child2, &c2, nullptr, 0, reinterpret_cast<void*>(gated_write), c2_args);
  add_kernel_node(child3, &c3, nullptr, 0, reinterpret_cast<void*>(write_val), c3_args);

  hipGraph_t parent;
  HIP_CHECK(hipGraphCreate(&parent, 0));
  hipGraphNode_t p_child1, p_child2, p_child3;
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child1, parent, nullptr, 0, child1));
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child2, parent, nullptr, 0, child2));
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child3, parent, nullptr, 0, child3));

  hipGraphNode_t deps[3] = {p_child1, p_child2, p_child3};
  void* sum_args[] = {&d_sum, &d_a, &d_b, &d_c};
  hipGraphNode_t p_sum;
  add_kernel_node(parent, &p_sum, deps, 3, reinterpret_cast<void*>(add_three_vals), sum_args);

  const std::string dot_path = PrepareDotFile();
  DotFileGuard dot_guard(dot_path);

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, parent, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipLaunchKernelGGL(gated_zero3, dim3(1), dim3(1), 0, stream, gs, kPreLaunchGate, d_a, d_b, d_c);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipGraphLaunch(exec, stream));
  gates.Release(kPreLaunchGate);
  gates.Release(kChildGate);
  HIP_CHECK(hipStreamSynchronize(stream));

  int sum = 0;
  HIP_CHECK(hipMemcpy(&sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE_FALSE(gates.TimedOut());
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

/**
 * parent: producer -> p_child(outer) -> p_consumer
 * outer:  o_child(inner) -> oz
 * inner:  ix, iy
 *
 * iy is an off-stream root of the inner child. It must wait on the producer
 * through two levels of child entry signals, and oz must wait for iy, which is
 * held on a gate.
 */
TEST_CASE("Unit_hipGraphChildMultiStream_NestedChild", "[graph][child][multistream][level_2]") {
  constexpr int kProducerGate = 0;
  constexpr int kInnerLeafGate = 1;

  HostGates gates;
  GateState* gs = gates.get();
  hipGraph_t inner, outer, parent;
  HIP_CHECK(hipGraphCreate(&inner, 0));
  HIP_CHECK(hipGraphCreate(&outer, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));

  int *d_p, *d_x, *d_y, *d_z, *d_zero, *d_final;
  HIP_CHECK(hipMalloc(&d_p, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_x, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_y, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_z, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_zero, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_final, sizeof(int)));
  HIP_CHECK(hipMemset(d_p, 0, sizeof(int)));
  HIP_CHECK(hipMemset(d_y, 0, sizeof(int)));
  HIP_CHECK(hipMemset(d_zero, 0, sizeof(int)));

  int one = 1, five = 5, seven = 7;
  int producer_gate = kProducerGate;
  int inner_leaf_gate = kInnerLeafGate;
  void* producer_args[] = {&gs, &producer_gate, &d_p, &one};
  void* ix_args[] = {&d_x, &five};
  void* iy_args[] = {&gs, &inner_leaf_gate, &d_p, &one, &d_y, &seven};
  void* oz_args[] = {&d_z, &d_x, &d_y};
  void* consumer_args[] = {&d_final, &d_z, &d_zero};

  hipGraphNode_t ix, iy;
  add_kernel_node(inner, &ix, nullptr, 0, reinterpret_cast<void*>(write_val), ix_args);
  add_kernel_node(inner, &iy, nullptr, 0, reinterpret_cast<void*>(gated_check_write), iy_args);

  hipGraphNode_t o_child, oz;
  HIP_CHECK(hipGraphAddChildGraphNode(&o_child, outer, nullptr, 0, inner));
  add_kernel_node(outer, &oz, &o_child, 1, reinterpret_cast<void*>(add_vals), oz_args);

  hipGraphNode_t producer, p_child, p_consumer;
  add_kernel_node(parent, &producer, nullptr, 0, reinterpret_cast<void*>(gated_write),
                  producer_args);
  HIP_CHECK(hipGraphAddChildGraphNode(&p_child, parent, &producer, 1, outer));
  add_kernel_node(parent, &p_consumer, &p_child, 1, reinterpret_cast<void*>(add_vals),
                  consumer_args);

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, parent, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  gates.Release(kProducerGate);
  gates.Release(kInnerLeafGate);
  HIP_CHECK(hipStreamSynchronize(stream));

  int final = 0;
  HIP_CHECK(hipMemcpy(&final, d_final, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE_FALSE(gates.TimedOut());
  REQUIRE(final == 12);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(parent));
  HIP_CHECK(hipGraphDestroy(outer));
  HIP_CHECK(hipGraphDestroy(inner));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_p));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_z));
  HIP_CHECK(hipFree(d_zero));
  HIP_CHECK(hipFree(d_final));
}
#endif
