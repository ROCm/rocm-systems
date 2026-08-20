/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Tests for hipGraphConditionalNode — WHILE, IF/IF-ELSE and SWITCH types.
 *
 * Validates:
 *   - hipGraphConditionalHandleCreate / hipGraphAddConditionalNode APIs
 *   - hipGraphSetConditional device function
 *   - GPU-side WHILE loop correctness across iteration counts
 *   - GPU-side WHILE with multiple body kernels
 *   - Zero-iteration WHILE (condition starts false)
 *   - GPU-side IF branch selection and IF-ELSE (2-body) branch selection
 *   - SWITCH case selection, out-of-range, and device-computed selectors
 *   - Nested conditionals: IF-in-WHILE, WHILE-in-IF, WHILE-in-WHILE
 *   - Multi-handle relaunch reset (two sequential top-level WHILE loops)
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

// --- Nested-conditional / IF-ELSE / SWITCH test kernels --------------------

// IF body: bump accum by 10 (counts how many times a nested IF ran).
static __global__ void ifAddKernel(float* accum) { *accum += 10.0f; }

// WHILE step that ALSO (re)arms a nested IF handle so the inner IF fires on
// every outer iteration. Exercises IF-inside-WHILE flattening.
static __global__ void whileStepArmIfKernel(hipGraphConditionalHandle hWhile,
                                            hipGraphConditionalHandle hIf,
                                            int* counter, const int* limit) {
  *counter += 1;
  hipGraphSetConditional(hWhile, (*counter < *limit) ? 1 : 0);
  hipGraphSetConditional(hIf, 1);
}

// Arm a nested WHILE at the top of each outer iteration: reset its counter and
// set its condition true so the inner loop runs afresh every time it's entered.
static __global__ void armWhileKernel(hipGraphConditionalHandle hInner,
                                      int* innerCounter) {
  *innerCounter = 0;
  hipGraphSetConditional(hInner, 1);
}

// Outer WHILE step: runs AFTER the inner loop completes; advances the outer
// counter and sets the outer condition. Exercises WHILE-inside-WHILE.
static __global__ void outerStepKernel(hipGraphConditionalHandle hOuter,
                                       int* outerCounter, const int* outerLimit) {
  *outerCounter += 1;
  hipGraphSetConditional(hOuter, (*outerCounter < *outerLimit) ? 1 : 0);
}

// Write a scalar value to *buf (used for IF-ELSE and SWITCH data validation).
static __global__ void writeValKernel(int* buf, int val) { *buf = val; }

// Compute a SWITCH selector into a handle from device data (device-set case).
static __global__ void setSwitchKernel(hipGraphConditionalHandle h, const int* sel) {
  hipGraphSetConditional(h, static_cast<unsigned>(*sel));
}

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
                                        &bodyGraph, 0));

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
                                        &bodyGraph, 0));

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

// Helper: build an IF graph with both true/false branch graphs, launch it,
// and verify the branch selected by defaultValue produced the expected output.
// hipGraphCondTypeIf requires exactly 2 conditional graphs (true, then false).
static void runIfTest(unsigned int defaultValue, int expectedOutput) {
  int* d_output;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));

  int zero = 0;
  HIP_CHECK(hipMemcpy(d_output, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, defaultValue, 0));

  hipGraph_t trueGraph = nullptr, falseGraph = nullptr;
  HIP_CHECK(hipGraphCreate(&trueGraph, 0));
  HIP_CHECK(hipGraphCreate(&falseGraph, 0));

  hipKernelNodeParams kpTrue = {};
  kpTrue.func = reinterpret_cast<void*>(ifTrueKernel);
  kpTrue.gridDim = dim3(1);
  kpTrue.blockDim = dim3(1);
  void* argsTrue[] = {&d_output};
  kpTrue.kernelParams = argsTrue;
  hipGraphNode_t trueNode;
  HIP_CHECK(hipGraphAddKernelNode(&trueNode, trueGraph, nullptr, 0, &kpTrue));

  hipKernelNodeParams kpFalse = {};
  kpFalse.func = reinterpret_cast<void*>(ifFalseKernel);
  kpFalse.gridDim = dim3(1);
  kpFalse.blockDim = dim3(1);
  void* argsFalse[] = {&d_output};
  kpFalse.kernelParams = argsFalse;
  hipGraphNode_t falseNode;
  HIP_CHECK(hipGraphAddKernelNode(&falseNode, falseGraph, nullptr, 0, &kpFalse));

  hipGraph_t branches[2] = {trueGraph, falseGraph};
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf, 2,
                                        branches, 0));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_output = 0;
  HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(h_output == expectedOutput);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
}

TEST_CASE("Unit_hipGraphConditionalNode_If_TrueBranch") {
  runIfTest(1, 42);
}

TEST_CASE("Unit_hipGraphConditionalNode_If_FalseBranch") {
  runIfTest(0, -1);
}

// ===========================================================================
// Nested conditionals
// ===========================================================================

// IF nested in WHILE: each WHILE iteration arms and runs a nested IF that bumps
// accum by 10. Verifies counter == numIters and accum == numIters*10, and that
// the whole nested structure re-arms correctly across launches.
static void runIfInsideWhileTest(int numIters) {
  int *d_counter, *d_limit;
  float* d_accum;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_limit, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_accum, sizeof(float)));
  HIP_CHECK(hipMemcpy(d_limit, &numIters, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle hWhile, hIf;
  HIP_CHECK(hipGraphConditionalHandleCreate(&hWhile, graph, 1, 0));
  HIP_CHECK(hipGraphConditionalHandleCreate(&hIf, graph, 0, 0));

  hipGraph_t bodyWhile = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyWhile, 0));
  hipGraphNode_t condWhile = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condWhile, graph, nullptr, 0, hWhile,
                                       hipGraphCondTypeWhile, 1, &bodyWhile, 0));

  hipKernelNodeParams sp = {};
  sp.gridDim = dim3(1);
  sp.blockDim = dim3(1);
  sp.func = reinterpret_cast<void*>(whileStepArmIfKernel);
  void* sargs[] = {&hWhile, &hIf, &d_counter, &d_limit};
  sp.kernelParams = sargs;
  hipGraphNode_t stepNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&stepNode, bodyWhile, nullptr, 0, &sp));

  hipGraph_t bodyIf = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyIf, 0));
  hipGraphNode_t condIf = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condIf, bodyWhile, &stepNode, 1, hIf,
                                       hipGraphCondTypeIf, 1, &bodyIf, 0));

  hipKernelNodeParams ap = {};
  ap.gridDim = dim3(1);
  ap.blockDim = dim3(1);
  ap.func = reinterpret_cast<void*>(ifAddKernel);
  void* aargs[] = {&d_accum};
  ap.kernelParams = aargs;
  hipGraphNode_t addNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&addNode, bodyIf, nullptr, 0, &ap));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  for (int run = 0; run < 2; ++run) {
    int zero = 0;
    float fzero = 0.0f;
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_accum, &fzero, sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int h_counter = -1;
    float h_accum = -1.0f;
    HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_accum, d_accum, sizeof(float), hipMemcpyDeviceToHost));
    REQUIRE(h_counter == numIters);
    REQUIRE(h_accum == Catch::Approx(static_cast<float>(numIters) * 10.0f).epsilon(0.001));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipFree(d_limit));
  HIP_CHECK(hipFree(d_accum));
}

TEST_CASE("Unit_hipGraphConditionalNode_IfInsideWhile_10iter") {
  runIfInsideWhileTest(10);
}

TEST_CASE("Unit_hipGraphConditionalNode_IfInsideWhile_100iter") {
  runIfInsideWhileTest(100);
}

// WHILE nested in IF. Outer IF taken iff takeBranch; its body holds a WHILE that
// runs innerIters times. Verifies the inner loop only runs when the branch is
// taken.
static void runWhileInsideIfTest(int innerIters, bool takeBranch) {
  int *d_counter, *d_limit;
  float* d_accum;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_limit, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_accum, sizeof(float)));
  HIP_CHECK(hipMemcpy(d_limit, &innerIters, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle hIf, hWhile;
  HIP_CHECK(hipGraphConditionalHandleCreate(&hIf, graph, takeBranch ? 1 : 0, 0));
  HIP_CHECK(hipGraphConditionalHandleCreate(&hWhile, graph, 1, 0));

  hipGraph_t bodyIf = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyIf, 0));
  hipGraphNode_t condIf = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condIf, graph, nullptr, 0, hIf,
                                       hipGraphCondTypeIf, 1, &bodyIf, 0));

  hipGraph_t bodyWhile = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyWhile, 0));
  hipGraphNode_t condWhile = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condWhile, bodyIf, nullptr, 0, hWhile,
                                       hipGraphCondTypeWhile, 1, &bodyWhile, 0));

  hipKernelNodeParams kp = {};
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  kp.func = reinterpret_cast<void*>(whileBodyKernel);
  void* args[] = {&hWhile, &d_counter, &d_limit, &d_accum};
  kp.kernelParams = args;
  hipGraphNode_t kNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyWhile, nullptr, 0, &kp));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  int zero = 0;
  float fzero = 0.0f;
  HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_accum, &fzero, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_counter = -1;
  HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(h_counter == (takeBranch ? innerIters : 0));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipFree(d_limit));
  HIP_CHECK(hipFree(d_accum));
}

TEST_CASE("Unit_hipGraphConditionalNode_WhileInsideIf_BranchTaken") {
  runWhileInsideIfTest(50, true);
}

TEST_CASE("Unit_hipGraphConditionalNode_WhileInsideIf_BranchNotTaken") {
  runWhileInsideIfTest(50, false);
}

// WHILE nested in WHILE (double loop). Outer runs outerIters; each outer
// iteration re-arms and runs an inner WHILE of innerIters. Verifies outer
// counter == outerIters and accum == outerIters*innerIters.
static void runWhileInsideWhileTest(int outerIters, int innerIters) {
  int *d_outer, *d_outerLim, *d_inner, *d_innerLim;
  float* d_accum;
  HIP_CHECK(hipMalloc(&d_outer, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_outerLim, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_inner, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_innerLim, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_accum, sizeof(float)));
  HIP_CHECK(hipMemcpy(d_outerLim, &outerIters, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_innerLim, &innerIters, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle hOuter, hInner;
  HIP_CHECK(hipGraphConditionalHandleCreate(&hOuter, graph, 1, 0));
  HIP_CHECK(hipGraphConditionalHandleCreate(&hInner, graph, 1, 0));

  hipGraph_t bodyOuter = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyOuter, 0));
  hipGraphNode_t condOuter = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condOuter, graph, nullptr, 0, hOuter,
                                       hipGraphCondTypeWhile, 1, &bodyOuter, 0));

  hipKernelNodeParams armp = {};
  armp.gridDim = dim3(1);
  armp.blockDim = dim3(1);
  armp.func = reinterpret_cast<void*>(armWhileKernel);
  void* armArgs[] = {&hInner, &d_inner};
  armp.kernelParams = armArgs;
  hipGraphNode_t armNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&armNode, bodyOuter, nullptr, 0, &armp));

  hipGraph_t bodyInner = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyInner, 0));
  hipGraphNode_t condInner = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condInner, bodyOuter, &armNode, 1, hInner,
                                       hipGraphCondTypeWhile, 1, &bodyInner, 0));

  hipKernelNodeParams ip = {};
  ip.gridDim = dim3(1);
  ip.blockDim = dim3(1);
  ip.func = reinterpret_cast<void*>(whileBodyKernel);
  void* iargs[] = {&hInner, &d_inner, &d_innerLim, &d_accum};
  ip.kernelParams = iargs;
  hipGraphNode_t innerBodyNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&innerBodyNode, bodyInner, nullptr, 0, &ip));

  hipKernelNodeParams op = {};
  op.gridDim = dim3(1);
  op.blockDim = dim3(1);
  op.func = reinterpret_cast<void*>(outerStepKernel);
  void* oargs[] = {&hOuter, &d_outer, &d_outerLim};
  op.kernelParams = oargs;
  hipGraphNode_t outerStepNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&outerStepNode, bodyOuter, &condInner, 1, &op));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  int zero = 0;
  float fzero = 0.0f;
  HIP_CHECK(hipMemcpy(d_outer, &zero, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_inner, &zero, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_accum, &fzero, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_outer = -1;
  float h_accum = -1.0f;
  HIP_CHECK(hipMemcpy(&h_outer, d_outer, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&h_accum, d_accum, sizeof(float), hipMemcpyDeviceToHost));
  REQUIRE(h_outer == outerIters);
  REQUIRE(h_accum == Catch::Approx(static_cast<float>(outerIters) *
                                   static_cast<float>(innerIters)).epsilon(0.001));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_outer));
  HIP_CHECK(hipFree(d_outerLim));
  HIP_CHECK(hipFree(d_inner));
  HIP_CHECK(hipFree(d_innerLim));
  HIP_CHECK(hipFree(d_accum));
}

TEST_CASE("Unit_hipGraphConditionalNode_WhileInsideWhile_5x10") {
  runWhileInsideWhileTest(5, 10);
}

TEST_CASE("Unit_hipGraphConditionalNode_WhileInsideWhile_10x20") {
  runWhileInsideWhileTest(10, 20);
}

// ===========================================================================
// IF-ELSE (2-body IF)
// ===========================================================================

// handle=1 runs the IF body (writes 42); handle=0 runs the ELSE body (99).
static void runIfElseTest(bool takeIf) {
  int* d_out;
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
  int init = -1;
  HIP_CHECK(hipMemcpy(d_out, &init, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphConditionalHandle h;
  HIP_CHECK(hipGraphConditionalHandleCreate(&h, graph, takeIf ? 1 : 0, 0));

  hipGraph_t ifBody = nullptr, elseBody = nullptr;
  HIP_CHECK(hipGraphCreate(&ifBody, 0));
  HIP_CHECK(hipGraphCreate(&elseBody, 0));
  hipGraph_t bodies[2] = {ifBody, elseBody};
  hipGraphNode_t condNode = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0, h,
                                       hipGraphCondTypeIf, 2, bodies, 0));

  int ifVal = 42, elseVal = 99;
  hipKernelNodeParams kp = {};
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  kp.func = reinterpret_cast<void*>(writeValKernel);
  void* ifArgs[] = {&d_out, &ifVal};
  kp.kernelParams = ifArgs;
  hipGraphNode_t ifNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&ifNode, ifBody, nullptr, 0, &kp));
  void* elseArgs[] = {&d_out, &elseVal};
  kp.kernelParams = elseArgs;
  hipGraphNode_t elseNode = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&elseNode, elseBody, nullptr, 0, &kp));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_out = 0;
  HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(h_out == (takeIf ? 42 : 99));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_out));
}

TEST_CASE("Unit_hipGraphConditionalNode_IfElse_TakeIf") {
  runIfElseTest(true);
}

TEST_CASE("Unit_hipGraphConditionalNode_IfElse_TakeElse") {
  runIfElseTest(false);
}

// ===========================================================================
// SWITCH
// ===========================================================================

// N case bodies; case c writes (c+1)*10. condVal selects the case; an
// out-of-range value runs no body and leaves the sentinel (-1). deviceSet=true
// routes the selector through an upstream kernel (device-computed).
static void runSwitchTest(int condVal, int numCases, bool deviceSet) {
  int *d_out, *d_sel;
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_sel, sizeof(int)));
  int sentinel = -1;
  HIP_CHECK(hipMemcpy(d_out, &sentinel, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sel, &condVal, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphConditionalHandle h;
  HIP_CHECK(hipGraphConditionalHandleCreate(&h, graph,
                                            deviceSet ? 0 : condVal, 0));

  hipGraphNode_t setNode = nullptr;
  if (deviceSet) {
    hipKernelNodeParams skp = {};
    skp.gridDim = dim3(1);
    skp.blockDim = dim3(1);
    skp.func = reinterpret_cast<void*>(setSwitchKernel);
    void* sArgs[] = {&h, &d_sel};
    skp.kernelParams = sArgs;
    HIP_CHECK(hipGraphAddKernelNode(&setNode, graph, nullptr, 0, &skp));
  }

  hipGraph_t bodies[8] = {};
  int vals[8] = {};
  for (int c = 0; c < numCases; ++c) {
    HIP_CHECK(hipGraphCreate(&bodies[c], 0));
    vals[c] = (c + 1) * 10;
    hipKernelNodeParams kp = {};
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(1);
    kp.func = reinterpret_cast<void*>(writeValKernel);
    void* args[] = {&d_out, &vals[c]};
    kp.kernelParams = args;
    hipGraphNode_t kn = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kn, bodies[c], nullptr, 0, &kp));
  }

  hipGraphNode_t condNode = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph,
                                       deviceSet ? &setNode : nullptr,
                                       deviceSet ? 1 : 0, h,
                                       hipGraphCondTypeSwitch, numCases,
                                       bodies, 0));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_out = 0;
  HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));
  const int expected =
      (condVal >= 0 && condVal < numCases) ? (condVal + 1) * 10 : -1;
  REQUIRE(h_out == expected);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_sel));
}

TEST_CASE("Unit_hipGraphConditionalNode_Switch_Case0") {
  runSwitchTest(0, 3, false);
}

TEST_CASE("Unit_hipGraphConditionalNode_Switch_Case1") {
  runSwitchTest(1, 3, false);
}

TEST_CASE("Unit_hipGraphConditionalNode_Switch_Case2") {
  runSwitchTest(2, 3, false);
}

TEST_CASE("Unit_hipGraphConditionalNode_Switch_OutOfRange") {
  runSwitchTest(5, 3, false);
}

TEST_CASE("Unit_hipGraphConditionalNode_Switch_DeviceComputed") {
  runSwitchTest(1, 3, true);
}

TEST_CASE("Unit_hipGraphConditionalNode_Switch_4Cases_DeviceComputed") {
  runSwitchTest(2, 4, true);
}

// ===========================================================================
// Multi-handle relaunch regression
// ===========================================================================

// Two INDEPENDENT top-level WHILE loops in sequence (B depends on A), each with
// its own handle. Regression for multi-handle launch reset: on the 2nd+ launch
// BOTH handles must be re-armed to their defaults, not just the first one.
static void runTwoSequentialWhileTest(int itersA, int itersB) {
  int *dCntA, *dLimA, *dCntB, *dLimB;
  float *dAccA, *dAccB;
  HIP_CHECK(hipMalloc(&dCntA, sizeof(int)));
  HIP_CHECK(hipMalloc(&dLimA, sizeof(int)));
  HIP_CHECK(hipMalloc(&dCntB, sizeof(int)));
  HIP_CHECK(hipMalloc(&dLimB, sizeof(int)));
  HIP_CHECK(hipMalloc(&dAccA, sizeof(float)));
  HIP_CHECK(hipMalloc(&dAccB, sizeof(float)));
  HIP_CHECK(hipMemcpy(dLimA, &itersA, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dLimB, &itersB, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphConditionalHandle hA, hB;
  HIP_CHECK(hipGraphConditionalHandleCreate(&hA, graph, 1, 0));
  HIP_CHECK(hipGraphConditionalHandleCreate(&hB, graph, 1, 0));

  hipGraph_t bodyA = nullptr, bodyB = nullptr;
  HIP_CHECK(hipGraphCreate(&bodyA, 0));
  HIP_CHECK(hipGraphCreate(&bodyB, 0));

  hipGraphNode_t condA = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condA, graph, nullptr, 0, hA,
                                       hipGraphCondTypeWhile, 1, &bodyA, 0));
  hipGraphNode_t condB = nullptr;
  HIP_CHECK(hipGraphAddConditionalNode(&condB, graph, &condA, 1, hB,
                                       hipGraphCondTypeWhile, 1, &bodyB, 0));

  hipKernelNodeParams kp = {};
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  kp.func = reinterpret_cast<void*>(whileBodyKernel);
  void* argsA[] = {&hA, &dCntA, &dLimA, &dAccA};
  kp.kernelParams = argsA;
  hipGraphNode_t knA = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&knA, bodyA, nullptr, 0, &kp));
  void* argsB[] = {&hB, &dCntB, &dLimB, &dAccB};
  kp.kernelParams = argsB;
  hipGraphNode_t knB = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&knB, bodyB, nullptr, 0, &kp));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  for (int run = 0; run < 3; ++run) {
    int zero = 0;
    float fzero = 0.0f;
    HIP_CHECK(hipMemcpy(dCntA, &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCntB, &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAccA, &fzero, sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAccB, &fzero, sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int hCntA = -1, hCntB = -1;
    HIP_CHECK(hipMemcpy(&hCntA, dCntA, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hCntB, dCntB, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(hCntA == itersA);
    REQUIRE(hCntB == itersB);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(dCntA));
  HIP_CHECK(hipFree(dLimA));
  HIP_CHECK(hipFree(dCntB));
  HIP_CHECK(hipFree(dLimB));
  HIP_CHECK(hipFree(dAccA));
  HIP_CHECK(hipFree(dAccB));
}

TEST_CASE("Unit_hipGraphConditionalNode_TwoSequentialWhile_10_20") {
  runTwoSequentialWhileTest(10, 20);
}

TEST_CASE("Unit_hipGraphConditionalNode_TwoSequentialWhile_7_3") {
  runTwoSequentialWhileTest(7, 3);
}
