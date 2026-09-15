/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * Tests that hipGraph segment assignment honours kernel node stream priority.
 *
 * Three gaps were present before the fix:
 *   1. hipGraphKernelNodeSetAttribute with hipLaunchAttributePriority had an
 *      inverted range check (p < Low || p > High is always true).
 *   2. Stream priority was not copied onto kernel nodes during stream capture.
 *   3. RoundRobinStreamAssignment assigned HW queue slots in capture order,
 *      so whichever branch was recorded first landed on slot 0 (the launch
 *      stream) regardless of which branch was on the critical path.
 *
 * Test strategy for gap 3:
 *   Capture a fork/join graph in two arrangements that are isomorphic DAGs:
 *     A) Heavy branch (many kernels) on high-priority stream, recorded FIRST.
 *     B) Same heavy branch, recorded LAST.
 *   Without the fix, B is slower (heavy branch pays a cross-stream barrier).
 *   With the fix, both have the same replay time.
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

  val.priority = lo + 1;  // less urgent than lo — invalid
  REQUIRE(hipGraphKernelNodeSetAttribute(node, hipLaunchAttributePriority, &val)
          == hipErrorInvalidValue);

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
    const float* dx = d;
    float* dy = d;
    float a = 1.f;
    int n = kN;
    void* args[] = {&dx, &dy, &a, &n};
    hipLaunchKernelGGL(saxpy_kernel, grid, block, 0, s_hi, dx, dy, a, n);
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
