/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipGraphAddHostNode hipGraphAddHostNode
 * @{
 * @ingroup GraphTest
 */

namespace {
__global__ void produceValue(int* dst, int value) { *dst = value; }
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Regression test for the GraphHostNode callback fence scope (clr:
 *      hipamd/src/hip_graph_internal.hpp). A host node's callback runs on the
 *      CPU and may read host memory produced by an upstream GPU node. The marker
 *      that gates the callback must carry a system-scope release fence so the
 *      producer's writes are flushed past L2 and are visible to the CPU before
 *      the callback executes.
 *
 *      Minimal graph (no multi-stream setup — the bug is stream-independent):
 *
 *        producerNode  write *devProduced = kProducedValue
 *            │
 *            ▼  (dependency)
 *        copyNode      devProduced → hostProduced   (captured DtoH memcpy)
 *            │
 *            ▼  (dependency)
 *        callbackNode  callback copies *hostProduced → *hostObserved
 *
 *      No spin/delay is needed: ordering is guaranteed by the dependency chain,
 *      and the visibility bug is deterministic on the callback marker's fence
 *      scope, not on timing (a NOP-scope release leaves the DtoH write invisible
 *      to the CPU regardless of how long the callback waits).
 *
 *      With the system-scope fence (fix): the DtoH write is CPU-visible, so the
 *      callback observes kProducedValue → hostObserved == kProducedValue ✓
 *      With a NOP-scope fence (bug): the callback reads un-flushed/stale memory
 *      on GPUs whose producer release is not host-visible (e.g. gfx1201)
 *      → hostObserved != kProducedValue  FAIL ✗
 *
 *      hostProduced is set to 0 on the host before launch, so if the DtoH write
 *      is not flushed the callback reads back the host's own cached 0
 *      (!= kProducedValue) and the test fails. The bug is deterministic, so a
 *      single launch suffices — no iteration loop is needed.
 * Test source
 * ------------------------
 *    - catch/unit/graph/hipGraphHostNodeCacheVisibility.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
HIP_TEST_CASE(Unit_hipGraphAddHostNode_CallbackSeesUpstreamDtoH) {
  constexpr int kProducedValue = 0x5A5A5A5A;

  int* hostProduced{};  // DtoH destination (GPU-produced host memory)
  int* hostObserved{};  // value the callback read back from hostProduced
  HIP_CHECK(hipHostMalloc(&hostProduced, sizeof(int), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&hostObserved, sizeof(int), hipHostMallocDefault));

  int* devProduced{};
  HIP_CHECK(hipMalloc(&devProduced, sizeof(int)));

  hipGraph_t hostFuncGraph{};
  HIP_CHECK(hipGraphCreate(&hostFuncGraph, 0));

  // Kernel node params are captured at instantiate; kProducedValue is a fixed
  // nonzero value baked into the graph.
  int producedValue = kProducedValue;
  void* producerArgs[] = {&devProduced, &producedValue};

  // producerNode: write kProducedValue into device memory.
  hipGraphNode_t producerNode{};
  {
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(produceValue);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = producerArgs;
    HIP_CHECK(hipGraphAddKernelNode(&producerNode, hostFuncGraph, nullptr, 0, &params));
  }

  // copyNode: captured DtoH devProduced → hostProduced, depends on producerNode.
  hipGraphNode_t copyNode{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&copyNode, hostFuncGraph, &producerNode, 1, hostProduced,
                                    devProduced, sizeof(int), hipMemcpyDeviceToHost));

  // callbackNode: uncaptured host callback reads the GPU-produced hostProduced.
  struct CallbackContext { int* produced; int* observed; };
  CallbackContext callbackContext{hostProduced, hostObserved};
  hipGraphNode_t callbackNode{};
  hipHostNodeParams callbackParams{};
  callbackParams.fn = [](void* userData) {
    auto* ctx = static_cast<CallbackContext*>(userData);
    *ctx->observed = *ctx->produced;
  };
  callbackParams.userData = &callbackContext;
  HIP_CHECK(hipGraphAddHostNode(&callbackNode, hostFuncGraph, &copyNode, 1, &callbackParams));

  hipGraphExec_t hostFuncGraphExec{};
  HIP_CHECK(hipGraphInstantiate(&hostFuncGraphExec, hostFuncGraph, nullptr, nullptr, 0));
  hipStream_t launchStream{};
  HIP_CHECK(hipStreamCreate(&launchStream));

  // Plant a known host-cached value: if the DtoH write is not flushed to be
  // CPU-visible, the callback reads back this 0 instead of kProducedValue.
  *hostProduced = 0;

  HIP_CHECK(hipGraphLaunch(hostFuncGraphExec, launchStream));
  HIP_CHECK(hipStreamSynchronize(launchStream));

  INFO("expected=" << kProducedValue << " observed=" << *hostObserved);
  REQUIRE(*hostObserved == kProducedValue);  // stale/un-flushed read → reads 0

  HIP_CHECK(hipGraphExecDestroy(hostFuncGraphExec));
  HIP_CHECK(hipGraphDestroy(hostFuncGraph));
  HIP_CHECK(hipStreamDestroy(launchStream));
  HIP_CHECK(hipFree(devProduced));
  HIP_CHECK(hipHostFree(hostObserved));
  HIP_CHECK(hipHostFree(hostProduced));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
