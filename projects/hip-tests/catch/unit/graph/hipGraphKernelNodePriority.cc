/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * Tests for hipGraph kernel node priority support:
 *   1. hipGraphKernelNodeSetAttribute/GetAttribute with hipLaunchAttributePriority
 *      accept values in [greatest, least] and reject anything outside it.
 *   2. Kernels captured on a prioritized stream carry that stream's priority.
 *   3. hipGraphKernelNodeCopyAttributes copies every attribute slot.
 *   4. Graphs instantiated with hipGraphInstantiateFlagUseNodePriority (flat
 *      fork/join and graphs containing child graphs) execute correctly. Queue
 *      slot ordering itself is a scheduling hint and is not observable via the
 *      public API, so these tests validate correctness, not placement.
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <vector>

namespace {

constexpr int kN = 1 << 20;

__global__ void saxpy_kernel(const float* __restrict__ x, float* __restrict__ y,
                             float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

bool AllEqual(const float* d_ptr, int n, float expected) {
  std::vector<float> h(n);
  HIP_CHECK(hipMemcpy(h.data(), d_ptr, n * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (h[i] != expected) return false;
  }
  return true;
}

void FillOnes(float* d_ptr, int n) {
  std::vector<float> h(n, 1.f);
  HIP_CHECK(hipMemcpy(d_ptr, h.data(), n * sizeof(float), hipMemcpyHostToDevice));
}

}  // namespace

/* --------------------------------------------------------------------------
 * Positive: SetAttribute with hipLaunchAttributePriority stores the value
 * and GetAttribute reads it back correctly.
 * -------------------------------------------------------------------------- */
HIP_TEST_CASE(Unit_hipGraphKernelNodeSetAttribute_Positive_Priority) {
  int lo, hi;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
  if (lo == hi) {
    HIP_SKIP_TEST("Device does not support stream priority");
  }

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kN * sizeof(float)));
  HIP_CHECK(hipMemset(d, 0, kN * sizeof(float)));

  hipKernelNodeParams p{};
  p.func = reinterpret_cast<void*>(saxpy_kernel);
  p.gridDim = dim3((kN + 255) / 256);
  p.blockDim = dim3(256);
  const float* dx = d;
  float* dy = d;
  float a = 1.f;
  int n = kN;
  void* args[] = {&dx, &dy, &a, &n};
  p.kernelParams = args;

  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &p));

  // Set high priority
  hipKernelNodeAttrValue val_set{};
  val_set.priority = hi;
  HIP_CHECK(hipGraphKernelNodeSetAttribute(node, hipLaunchAttributePriority, &val_set));

  // Read back and verify
  hipKernelNodeAttrValue val_get{};
  HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributePriority, &val_get));
  REQUIRE(val_get.priority == hi);

  // Set low priority and verify
  val_set.priority = lo;
  HIP_CHECK(hipGraphKernelNodeSetAttribute(node, hipLaunchAttributePriority, &val_set));
  HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributePriority, &val_get));
  REQUIRE(val_get.priority == lo);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d));
}

/* --------------------------------------------------------------------------
 * Negative: out-of-range priority values must return hipErrorInvalidValue.
 * -------------------------------------------------------------------------- */
HIP_TEST_CASE(Unit_hipGraphKernelNodeSetAttribute_Negative_Priority_OutOfRange) {
  int lo, hi;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(float)));
  hipKernelNodeParams p{};
  p.func = reinterpret_cast<void*>(saxpy_kernel);
  p.gridDim = dim3(1); p.blockDim = dim3(1);
  const float* dx = d; float* dy = d; float a = 1.f; int n = 1;
  void* args[] = {&dx, &dy, &a, &n};
  p.kernelParams = args;
  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &p));

  hipKernelNodeAttrValue val{};

  // Values outside [hi, lo] must be rejected.
  val.priority = hi - 1;  // more urgent than hi — invalid
  REQUIRE(hipGraphKernelNodeSetAttribute(node, hipLaunchAttributePriority, &val)
          == hipErrorInvalidValue);
  (void)hipGetLastError();  // clear sticky per-thread error

  val.priority = lo + 1;  // less urgent than lo — invalid
  REQUIRE(hipGraphKernelNodeSetAttribute(node, hipLaunchAttributePriority, &val)
          == hipErrorInvalidValue);
  (void)hipGetLastError();  // clear sticky per-thread error

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d));
}

/* --------------------------------------------------------------------------
 * Positive: stream priority is copied to kernel nodes during stream capture.
 *
 * Captures a kernel on a high-priority stream, then reads the priority back
 * from the graph node via hipGraphKernelNodeGetAttribute. Verifies that the
 * node carries the stream's priority, not the default Normal priority.
 * -------------------------------------------------------------------------- */
HIP_TEST_CASE(Unit_hipGraphKernelNodePriority_CaptureStreamPriorityCopied) {
  int lo, hi;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
  if (lo == hi) {
    HIP_SKIP_TEST("Device does not support stream priority");
  }

  float* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kN * sizeof(float)));
  HIP_CHECK(hipMemset(d, 0, kN * sizeof(float)));

  // Capture stream with high priority
  hipStream_t s_hi;
  HIP_CHECK(hipStreamCreateWithPriority(&s_hi, hipStreamDefault, hi));

  hipGraph_t graph;
  HIP_CHECK(hipStreamBeginCapture(s_hi, hipStreamCaptureModeGlobal));
  {
    dim3 block(256), grid((kN + 255) / 256);
    hipLaunchKernelGGL(saxpy_kernel, grid, block, 0, s_hi, d, d, 1.f, kN);
  }
  HIP_CHECK(hipStreamEndCapture(s_hi, &graph));

  // Walk graph nodes, find the kernel node, check its priority
  size_t num_nodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &num_nodes));
  REQUIRE(num_nodes > 0);

  std::vector<hipGraphNode_t> nodes(num_nodes);
  HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &num_nodes));

  bool found_kernel = false;
  for (auto node : nodes) {
    hipGraphNodeType type;
    HIP_CHECK(hipGraphNodeGetType(node, &type));
    if (type != hipGraphNodeTypeKernel) continue;

    hipKernelNodeAttrValue val{};
    HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributePriority, &val));

    // The node must carry the stream's priority, not the default Normal (0).
    INFO("Captured node priority: " << val.priority << ", stream priority: " << hi);
    CHECK(val.priority == hi);
    found_kernel = true;
  }

  REQUIRE(found_kernel);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(s_hi));
  HIP_CHECK(hipFree(d));
}

/* --------------------------------------------------------------------------
 * Positive: hipGraphKernelNodeCopyAttributes copies all attribute slots
 * (accessPolicyWindow, cooperative, priority) independently.
 *
 * Before the union->per-slot refactor, CopyAttr copied only the last-written
 * slot, so a node with two attributes set would lose all but one.
 * -------------------------------------------------------------------------- */
HIP_TEST_CASE(Unit_hipGraphKernelNodeCopyAttributes_MultiAttr) {
  int lo, hi;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
  if (lo == hi) {
    HIP_SKIP_TEST("Device does not support stream priority");
  }

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(float)));

  hipKernelNodeParams p{};
  p.func = reinterpret_cast<void*>(saxpy_kernel);
  p.gridDim = dim3(1); p.blockDim = dim3(1);
  const float* dx = d; float* dy = d; float a = 1.f; int n = 1;
  void* args[] = {&dx, &dy, &a, &n};
  p.kernelParams = args;

  hipGraphNode_t src, dst;
  HIP_CHECK(hipGraphAddKernelNode(&src, graph, nullptr, 0, &p));
  HIP_CHECK(hipGraphAddKernelNode(&dst, graph, nullptr, 0, &p));

  // Set cooperative on src.
  hipKernelNodeAttrValue coop{};
  coop.cooperative = 1;
  HIP_CHECK(hipGraphKernelNodeSetAttribute(src, hipKernelNodeAttributeCooperative, &coop));

  // Set priority on src.
  hipKernelNodeAttrValue prio{};
  prio.priority = hi;
  HIP_CHECK(hipGraphKernelNodeSetAttribute(src, hipLaunchAttributePriority, &prio));

  // Copy all attributes from src to dst.
  HIP_CHECK(hipGraphKernelNodeCopyAttributes(src, dst));

  // Both cooperative and priority must survive the copy.
  hipKernelNodeAttrValue got{};
  HIP_CHECK(hipGraphKernelNodeGetAttribute(dst, hipKernelNodeAttributeCooperative, &got));
  REQUIRE(got.cooperative == 1);

  HIP_CHECK(hipGraphKernelNodeGetAttribute(dst, hipLaunchAttributePriority, &got));
  REQUIRE(got.priority == hi);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d));
}

/* --------------------------------------------------------------------------
 * Positive: a fork/join graph instantiated with hipGraphInstantiateFlagUseNodePriority
 * runs to completion and produces correct output.
 *
 * Two branches capture saxpy on a high-priority and a low-priority stream, each
 * writing its own output buffer. Verifies the flag is accepted on the segmented
 * path and that both branches produce the expected values.
 * -------------------------------------------------------------------------- */
HIP_TEST_CASE(Unit_hipGraphKernelNodePriority_ForkJoinWithFlag) {
  int lo, hi;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
  if (lo == hi) {
    HIP_SKIP_TEST("Device does not support stream priority");
  }

  constexpr int kElems = 1 << 16;
  float* d_x = nullptr;
  float* d_y_hi = nullptr;
  float* d_y_lo = nullptr;
  HIP_CHECK(hipMalloc(&d_x, kElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_hi, kElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_lo, kElems * sizeof(float)));
  FillOnes(d_x, kElems);
  HIP_CHECK(hipMemset(d_y_hi, 0, kElems * sizeof(float)));
  HIP_CHECK(hipMemset(d_y_lo, 0, kElems * sizeof(float)));

  hipStream_t s_main, s_hi, s_lo;
  HIP_CHECK(hipStreamCreate(&s_main));
  HIP_CHECK(hipStreamCreateWithPriority(&s_hi, hipStreamDefault, hi));
  HIP_CHECK(hipStreamCreateWithPriority(&s_lo, hipStreamDefault, lo));

  // Fork: s_main -> s_hi and s_main -> s_lo; both join back to s_main.
  hipEvent_t fork_hi, fork_lo, join_hi, join_lo;
  HIP_CHECK(hipEventCreate(&fork_hi));
  HIP_CHECK(hipEventCreate(&fork_lo));
  HIP_CHECK(hipEventCreate(&join_hi));
  HIP_CHECK(hipEventCreate(&join_lo));

  hipGraph_t graph;
  HIP_CHECK(hipStreamBeginCapture(s_main, hipStreamCaptureModeGlobal));

  HIP_CHECK(hipEventRecord(fork_hi, s_main));
  HIP_CHECK(hipEventRecord(fork_lo, s_main));
  HIP_CHECK(hipStreamWaitEvent(s_hi, fork_hi, 0));
  HIP_CHECK(hipStreamWaitEvent(s_lo, fork_lo, 0));

  // Record the low-priority branch first so capture order disagrees with priority.
  dim3 block(256), grid((kElems + 255) / 256);
  hipLaunchKernelGGL(saxpy_kernel, grid, block, 0, s_lo, d_x, d_y_lo, 3.f, kElems);
  hipLaunchKernelGGL(saxpy_kernel, grid, block, 0, s_hi, d_x, d_y_hi, 2.f, kElems);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(join_hi, s_hi));
  HIP_CHECK(hipEventRecord(join_lo, s_lo));
  HIP_CHECK(hipStreamWaitEvent(s_main, join_hi, 0));
  HIP_CHECK(hipStreamWaitEvent(s_main, join_lo, 0));

  HIP_CHECK(hipStreamEndCapture(s_main, &graph));

  hipGraphExec_t exec;
  hipError_t inst_status = hipGraphInstantiateWithFlags(
      &exec, graph, hipGraphInstantiateFlagUseNodePriority);

  auto cleanup = [&]() {
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipEventDestroy(join_lo));
    HIP_CHECK(hipEventDestroy(join_hi));
    HIP_CHECK(hipEventDestroy(fork_lo));
    HIP_CHECK(hipEventDestroy(fork_hi));
    HIP_CHECK(hipStreamDestroy(s_lo));
    HIP_CHECK(hipStreamDestroy(s_hi));
    HIP_CHECK(hipStreamDestroy(s_main));
    HIP_CHECK(hipFree(d_y_lo));
    HIP_CHECK(hipFree(d_y_hi));
    HIP_CHECK(hipFree(d_x));
  };

  if (inst_status == hipErrorNotSupported) {
    // Clear the sticky per-thread error before skipping.
    (void)hipGetLastError();
    cleanup();
    HIP_SKIP_TEST("hipGraphInstantiateFlagUseNodePriority not supported on this backend");
  }
  HIP_CHECK(inst_status);

  HIP_CHECK(hipGraphLaunch(exec, s_main));
  HIP_CHECK(hipStreamSynchronize(s_main));

  // y = a * 1 + 0
  CHECK(AllEqual(d_y_hi, kElems, 2.f));
  CHECK(AllEqual(d_y_lo, kElems, 3.f));

  HIP_CHECK(hipGraphExecDestroy(exec));
  cleanup();
}

/* --------------------------------------------------------------------------
 * Positive: a graph containing child graphs instantiated with
 * hipGraphInstantiateFlagUseNodePriority executes correctly.
 *
 * Exercises the child-graph path of segment priority aggregation: one child
 * graph holds a kernel set to high priority, the other a kernel left at the
 * default. The parent runs both children in parallel and a final kernel that
 * depends on both.
 * -------------------------------------------------------------------------- */
HIP_TEST_CASE(Unit_hipGraphKernelNodePriority_ChildGraphWithFlag) {
  int lo, hi;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
  if (lo == hi) {
    HIP_SKIP_TEST("Device does not support stream priority");
  }

  constexpr int kElems = 1 << 16;
  float* d_x = nullptr;
  float* d_y_a = nullptr;
  float* d_y_b = nullptr;
  HIP_CHECK(hipMalloc(&d_x, kElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_a, kElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_b, kElems * sizeof(float)));
  FillOnes(d_x, kElems);
  HIP_CHECK(hipMemset(d_y_a, 0, kElems * sizeof(float)));
  HIP_CHECK(hipMemset(d_y_b, 0, kElems * sizeof(float)));

  const float* dx = d_x;
  float* dy_a = d_y_a;
  float* dy_b = d_y_b;
  float a_hi = 2.f, a_def = 3.f, a_tail = 4.f;
  int n = kElems;
  void* args_hi[] = {&dx, &dy_a, &a_hi, &n};
  void* args_def[] = {&dx, &dy_b, &a_def, &n};
  void* args_tail[] = {&dx, &dy_a, &a_tail, &n};

  auto make_params = [&](void** args) {
    hipKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(saxpy_kernel);
    p.gridDim = dim3((kElems + 255) / 256);
    p.blockDim = dim3(256);
    p.kernelParams = args;
    return p;
  };

  // Child A: one high-priority kernel.
  hipGraph_t child_a;
  HIP_CHECK(hipGraphCreate(&child_a, 0));
  hipGraphNode_t k_hi;
  hipKernelNodeParams p_hi = make_params(args_hi);
  HIP_CHECK(hipGraphAddKernelNode(&k_hi, child_a, nullptr, 0, &p_hi));
  hipKernelNodeAttrValue prio{};
  prio.priority = hi;
  HIP_CHECK(hipGraphKernelNodeSetAttribute(k_hi, hipLaunchAttributePriority, &prio));

  // Child B: one kernel at default priority.
  hipGraph_t child_b;
  HIP_CHECK(hipGraphCreate(&child_b, 0));
  hipGraphNode_t k_def;
  hipKernelNodeParams p_def = make_params(args_def);
  HIP_CHECK(hipGraphAddKernelNode(&k_def, child_b, nullptr, 0, &p_def));

  // Parent: child_b and child_a in parallel (low-priority branch added first),
  // then a tail kernel depending on both.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t n_b, n_a, n_tail;
  HIP_CHECK(hipGraphAddChildGraphNode(&n_b, graph, nullptr, 0, child_b));
  HIP_CHECK(hipGraphAddChildGraphNode(&n_a, graph, nullptr, 0, child_a));
  hipGraphNode_t deps[] = {n_a, n_b};
  hipKernelNodeParams p_tail = make_params(args_tail);
  HIP_CHECK(hipGraphAddKernelNode(&n_tail, graph, deps, 2, &p_tail));

  hipGraphExec_t exec;
  hipError_t inst_status = hipGraphInstantiateWithFlags(
      &exec, graph, hipGraphInstantiateFlagUseNodePriority);

  auto cleanup = [&]() {
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphDestroy(child_b));
    HIP_CHECK(hipGraphDestroy(child_a));
    HIP_CHECK(hipFree(d_y_b));
    HIP_CHECK(hipFree(d_y_a));
    HIP_CHECK(hipFree(d_x));
  };

  if (inst_status == hipErrorNotSupported) {
    (void)hipGetLastError();
    cleanup();
    HIP_SKIP_TEST("hipGraphInstantiateFlagUseNodePriority not supported on this backend");
  }
  HIP_CHECK(inst_status);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // d_y_a = 2*1 + 0, then tail adds 4*1 -> 6. d_y_b = 3*1 + 0.
  CHECK(AllEqual(d_y_a, kElems, 6.f));
  CHECK(AllEqual(d_y_b, kElems, 3.f));

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipStreamDestroy(stream));
  cleanup();
}
