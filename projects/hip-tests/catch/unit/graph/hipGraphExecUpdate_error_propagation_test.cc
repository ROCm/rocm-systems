/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphExecUpdate hipGraphExecUpdate
 * @{
 * @ingroup GraphTest
 * Regression test for the hipGraphExecUpdate error-propagation contract under
 * kernel-argument exhaustion (companion to
 * https://github.com/ROCm/rocm-systems/issues/10021: the bulk update loop
 * previously assigned the UpdateAQLPacket failure to a local and returned
 * hipSuccess, leaving the previously captured packet to launch).
 */

#include <hip_test_common.hh>

#include <vector>

namespace {
__global__ void WriteValue(int* out, int value) { *out = value; }
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Exhausts device memory, then repeatedly modifies a one-kernel graph and
 *    calls hipGraphExecUpdate + hipGraphLaunch until the update path fails.
 *    The contract under test:
 *      1. The failure MUST surface from hipGraphExecUpdate itself as
 *         hipErrorGraphExecUpdateFailure, with hErrorNode_out and
 *         updateResult_out set — never as hipSuccess (the propagation defect:
 *         a stale kernel-argument packet then launches silently).
 *      2. Every iteration whose calls all report success must produce the
 *         freshly-updated value (no stale packet).
 *      3. Destroying and re-instantiating the exec after the failure must
 *         recover: the rebuild reclaims kernarg pool space, and a subsequent
 *         update + launch succeeds and verifies (the framework fallback
 *         pattern, e.g. ggml's destroy+reinstantiate on this error code).
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate_error_propagation_test.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_ExhaustionErrorPropagation) {
  int* d_out{nullptr};
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int value = 0;
  void* kernelArgs[] = {&d_out, &value};
  hipKernelNodeParams nodeParams{};
  nodeParams.func = reinterpret_cast<void*>(WriteValue);
  nodeParams.gridDim = dim3(1);
  nodeParams.blockDim = dim3(1);
  nodeParams.sharedMemBytes = 0;
  nodeParams.kernelParams = kernelArgs;
  nodeParams.extra = nullptr;
  hipGraphNode_t kNode{};
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &nodeParams));

  hipGraphExec_t graphExec{};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  value = 1;
  HIP_CHECK(hipGraphLaunch(graphExec, nullptr));
  HIP_CHECK(hipStreamSynchronize(nullptr));

  // Exhaust device memory: greedy halving-probe allocation, keeping every block.
  std::vector<void*> hogs;
  size_t chunk = size_t{1} << 30;
  while (chunk >= (size_t{4} << 10)) {
    void* p{nullptr};
    if (hipMalloc(&p, chunk) == hipSuccess) {
      hogs.push_back(p);
    } else {
      chunk >>= 1;
    }
  }
  static_cast<void>(hipGetLastError());  // clear the expected final OOM

  // Drive updates until the update path fails (budget as in
  // hipGraphExecUpdate_oom_test.cc; observed on a 24 GB gfx1100 well inside it).
  constexpr int kIters = 262144;
  bool sawUpdateFailure = false;
  int lastGood = 1;
  for (int i = 2; i <= kIters; ++i) {
    value = i;
    HIP_CHECK(hipGraphKernelNodeSetParams(kNode, &nodeParams));
    hipGraphNode_t errorNode{};
    hipGraphExecUpdateResult updateResult{};
    const hipError_t status =
        hipGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    if (status != hipSuccess) {
      // Contract 1: the exhaustion failure surfaces as the update-failure
      // code with the out-parameters populated.
      REQUIRE(status == hipErrorGraphExecUpdateFailure);
      REQUIRE(errorNode != nullptr);
      REQUIRE(updateResult == hipGraphExecUpdateError);
      sawUpdateFailure = true;
      break;
    }
    hipError_t rc = hipGraphLaunch(graphExec, nullptr);
    if (rc == hipSuccess) rc = hipStreamSynchronize(nullptr);
    if (rc != hipSuccess) break;  // exhaustion may also surface here: clean, acceptable
    int seen = -1;
    HIP_CHECK(hipMemcpy(&seen, d_out, sizeof(int), hipMemcpyDeviceToHost));
    // Contract 2: success means the NEW packet ran (no stale kernel args).
    REQUIRE(seen == i);
    lastGood = i;
  }

  if (sawUpdateFailure) {
    // Contract 3: destroy + re-instantiate recovers (the rebuild reclaims
    // the exec's kernarg pool), and the next update + launch verifies.
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    graphExec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    value = lastGood + 1;
    HIP_CHECK(hipGraphKernelNodeSetParams(kNode, &nodeParams));
    hipGraphNode_t errorNode{};
    hipGraphExecUpdateResult updateResult{};
    HIP_CHECK(hipGraphExecUpdate(graphExec, graph, &errorNode, &updateResult));
    HIP_CHECK(hipGraphLaunch(graphExec, nullptr));
    HIP_CHECK(hipStreamSynchronize(nullptr));
    int seen = -1;
    HIP_CHECK(hipMemcpy(&seen, d_out, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(seen == lastGood + 1);
  }

  for (void* p : hogs) {
    static_cast<void>(hipFree(p));
  }
  static_cast<void>(hipGraphExecDestroy(graphExec));
  static_cast<void>(hipGraphDestroy(graph));
  static_cast<void>(hipFree(d_out));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
