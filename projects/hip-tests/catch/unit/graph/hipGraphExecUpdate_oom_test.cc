/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphExecUpdate hipGraphExecUpdate
 * @{
 * @ingroup GraphTest
 * Regression test for kernel-argument pool growth under device-memory
 * exhaustion (https://github.com/ROCm/rocm-systems/issues/10021).
 */

#include <hip_test_common.hh>

#include <vector>

namespace {
__global__ void WriteValue(int* out, int value) { *out = value; }
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Repeatedly updates and launches an executable graph while device memory
 *    is exhausted. Graph exec updates re-capture kernel packets and allocate
 *    fresh kernel-argument space on every update; once the pre-sized kernarg
 *    pool is consumed, pool growth allocates device memory and fails when the
 *    device is out of memory. That failure must surface as an error from the
 *    API (and the previously-launched packet must never be silently reused):
 *    prior to the fix for issue #10021 this path either crashed with SIGSEGV
 *    inside libamdhip64 (NULL-destination nontemporalMemcpy) or, with the
 *    null check alone, reported success while launching a stale kernel-arg
 *    packet (silent output corruption). The probe kernel writes its argument
 *    to device memory so stale-packet corruption is detected as a value
 *    mismatch.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate_oom_test.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_KernargPoolExhaustion) {
  int* d_out{nullptr};
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

  // One-kernel graph: kernel writes its integer argument to d_out.
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

  // Prime one launch while memory is still available.
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

  // Update + launch until either the iteration budget is reached or the
  // runtime reports a clean out-of-memory failure. Each update captures a
  // fresh kernel-argument slot; the budget is sized so the loop consumes the
  // pre-sized kernarg pool plus any residual device memory through pool
  // growth (observed trigger on a 24 GB gfx1100: ~135k iterations, ~2 min).
  constexpr int kIters = 262144;
  hipError_t status = hipSuccess;
  for (int i = 2; i <= kIters; ++i) {
    value = i;
    HIP_CHECK(hipGraphKernelNodeSetParams(kNode, &nodeParams));
    hipGraphNode_t errorNode{};
    hipGraphExecUpdateResult updateResult{};
    status = hipGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    if (status != hipSuccess) break;
    status = hipGraphLaunch(graphExec, nullptr);
    if (status != hipSuccess) break;
    status = hipStreamSynchronize(nullptr);
    if (status != hipSuccess) break;
    int seen = -1;
    status = hipMemcpy(&seen, d_out, sizeof(int), hipMemcpyDeviceToHost);
    if (status != hipSuccess) break;
    // A stale kernel-argument packet (issue #10021, propagation half) shows up
    // as the previous iteration's value here.
    REQUIRE(seen == i);
  }

  // Pass condition: reaching this point (no SIGSEGV) with no stale-packet
  // value mismatch on any iteration whose calls all reported success. Under
  // full exhaustion the runtime is permitted to fail any of the calls above
  // with a clean error code — the regression under test is the process crash
  // (issue #10021) and the silent stale-packet corruption, not which error
  // class eventually surfaces.
  static_cast<void>(status);

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
