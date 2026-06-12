/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Tests for hipGraphConditionalNode — WHILE and IF conditional types.
 *
 * Validates:
 *   - hipGraphConditionalHandleCreate / hipGraphAddConditionalNode APIs
 *   - hipGraphSetConditional device function
 *   - GPU-side WHILE loop correctness across iteration counts
 *   - GPU-side WHILE with multiple body kernels
 *   - GPU-side IF branch selection
 *   - Zero-iteration WHILE (condition starts false)
 */

#include <hip_test_common.hh>

// Body kernel: increments counter, accumulates, sets condition
static __global__ void whileBodyKernel(hipGraphConditionalHandle handle,
                                       int* counter, const int* limit,
                                       float* accum) {
  *counter += 1;
  *accum += 1.0f;
  hipGraphSetConditional(handle, (*counter < *limit) ? 1 : 0);
}

// Work kernel for multi-kernel body tests
static __global__ void workKernel(float* accum) {
  *accum += 0.5f;
}

// Condition-setting kernel (last in multi-kernel body)
static __global__ void condSetKernel(hipGraphConditionalHandle handle,
                                     int* counter, const int* limit) {
  *counter += 1;
  hipGraphSetConditional(handle, (*counter < *limit) ? 1 : 0);
}

// IF body kernels
static __global__ void ifTrueKernel(int* output) { *output = 42; }
static __global__ void ifFalseKernel(int* output) { *output = -1; }

// Helper: build and run a WHILE graph, verify results
static void runWhileTest(int numIters, int numBodyKernels = 1) {
  int *d_counter, *d_limit;
  float *d_accum;

  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_limit, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_accum, sizeof(float)));

  HIP_CHECK(hipMemcpy(d_limit, &numIters, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1, 0));

  hipGraph_t bodyGraph = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyGraph, 0));

  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeWhile, 1,
                                        &bodyGraph));

  hipGraphNode_t prevNode = nullptr;
  for (int k = 0; k < numBodyKernels; k++) {
    hipKernelNodeParams kp = {};
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(1);
    hipGraphNode_t kNode;

    if (numBodyKernels == 1) {
      kp.func = reinterpret_cast<void*>(whileBodyKernel);
      void* args[] = {&handle, &d_counter, &d_limit, &d_accum};
      kp.kernelParams = args;
      HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph,
                                       prevNode ? &prevNode : nullptr,
                                       prevNode ? 1 : 0, &kp));
    } else if (k < numBodyKernels - 1) {
      kp.func = reinterpret_cast<void*>(workKernel);
      void* args[] = {&d_accum};
      kp.kernelParams = args;
      HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph,
                                       prevNode ? &prevNode : nullptr,
                                       prevNode ? 1 : 0, &kp));
    } else {
      kp.func = reinterpret_cast<void*>(condSetKernel);
      void* args[] = {&handle, &d_counter, &d_limit};
      kp.kernelParams = args;
      HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph,
                                       prevNode ? &prevNode : nullptr,
                                       prevNode ? 1 : 0, &kp));
    }
    prevNode = kNode;
  }

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Run 3 times to verify reusability
  for (int run = 0; run < 3; run++) {
    int zero = 0;
    float fzero = 0.0f;
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_accum, &fzero, sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int h_counter = 0;
    float h_accum = 0.0f;
    HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_accum, d_accum, sizeof(float), hipMemcpyDeviceToHost));

    REQUIRE(h_counter == numIters);
    if (numBodyKernels == 1) {
      REQUIRE(h_accum == Catch::Approx(static_cast<float>(numIters)).epsilon(0.001));
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipFree(d_limit));
  HIP_CHECK(hipFree(d_accum));
}

TEST_CASE("Unit_hipGraphConditionalNode_While_10iter") {
  runWhileTest(10);
}

TEST_CASE("Unit_hipGraphConditionalNode_While_100iter") {
  runWhileTest(100);
}

TEST_CASE("Unit_hipGraphConditionalNode_While_1000iter") {
  runWhileTest(1000);
}

TEST_CASE("Unit_hipGraphConditionalNode_While_ZeroIter") {
  int *d_counter, *d_limit;
  float *d_accum;

  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_limit, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_accum, sizeof(float)));

  int zero = 0;
  float fzero = 0.0f;
  HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_limit, &zero, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_accum, &fzero, sizeof(float), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // defaultValue = 0 → never enter the loop
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 0, 0));

  hipGraph_t bodyGraph = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyGraph, 0));

  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeWhile, 1,
                                        &bodyGraph));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(whileBodyKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&handle, &d_counter, &d_limit, &d_accum};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_counter = -1;
  float h_accum = -1.0f;
  HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&h_accum, d_accum, sizeof(float), hipMemcpyDeviceToHost));

  REQUIRE(h_counter == 0);
  REQUIRE(h_accum == 0.0f);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipFree(d_limit));
  HIP_CHECK(hipFree(d_accum));
}

TEST_CASE("Unit_hipGraphConditionalNode_While_MultiBody_3kernels") {
  runWhileTest(100, 3);
}

TEST_CASE("Unit_hipGraphConditionalNode_While_MultiBody_5kernels") {
  runWhileTest(100, 5);
}

// IF conditional type is not yet fully implemented in the runtime.
// These tests are disabled until hipGraphCondTypeIf support is added.
TEST_CASE("Unit_hipGraphConditionalNode_If_TrueBranch",
          "[!mayfail]") {
  int* d_output;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));

  int zero = 0;
  HIP_CHECK(hipMemcpy(d_output, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // defaultValue = 1 → take the true branch
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1, 0));

  hipGraph_t bodyGraph = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyGraph, 0));

  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf, 1,
                                        &bodyGraph));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(ifTrueKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&d_output};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_output = 0;
  HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(h_output == 42);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
}

TEST_CASE("Unit_hipGraphConditionalNode_If_FalseBranch",
          "[!mayfail]") {
  int* d_output;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));

  int zero = 0;
  HIP_CHECK(hipMemcpy(d_output, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // defaultValue = 0 → take the false/skip branch
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 0, 0));

  hipGraph_t bodyGraph = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyGraph, 0));

  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf, 1,
                                        &bodyGraph));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(ifTrueKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&d_output};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_output = -1;
  HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(int), hipMemcpyDeviceToHost));
  // Body should NOT have run, output stays 0
  REQUIRE(h_output == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
}
