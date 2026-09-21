/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipGraphAddBatchMemOpNode hipGraphAddBatchMemOpNode
 * @{
 * @ingroup GraphTest
 * `hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t *phGraphNode, hipGraph_t
 hGraph, const hipGraphNode_t *dependencies, size_t numDependencies, const
 hipBatchMemOpNodeParams* nodeParams);`
 * - Creates a batch memory operation node and adds it to a graph
 */

/**
 * Test Description
 * ------------------------
 * - Verify that hipGraphBatchMemOpNode with WriteValue32 + WaitValue32 produces the
 *   correct result after hipGraphInstantiate and hipGraphLaunch.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphBatchMemOpNodeCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipGraphBatchMemOpNode_WriteAndWait) {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  // StreamWaitValue / SignalMemory not supported on all backends (e.g. Windows PAL)
  int waitValueSupport = 0;
  auto attrErr = hipDeviceGetAttribute(&waitValueSupport,
                                       hipDeviceAttributeCanUseStreamWaitValue, 0);
  if (attrErr != hipSuccess || waitValueSupport == 0) {
    HIP_SKIP_TEST("hipStreamWaitValue is not supported on this device.");
  }
#if !HT_AMD
  // hipMallocSignalMemory is AMD/ROCr-only; skip at runtime on non-AMD backends
  // so that devPtr=0 is never passed to hipMemset/hipMemcpy.
  HIP_SKIP_TEST("hipMallocSignalMemory is not supported on non-AMD backends.");
#endif
  hipCtx_t ctx;
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  // Allocate signal memory required for wait-value operations on AMD
  hipDeviceptr_t devPtr = 0;
#if HT_AMD
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtr), sizeof(uint64_t),
                                  hipMallocSignalMemory));
#endif
  HIP_CHECK(hipMemset(reinterpret_cast<void*>(devPtr), 0, sizeof(uint64_t)));

  // Build batch: WriteValue32(1000) then WaitValue32(== 1000)
  hipStreamBatchMemOpParams params[2] = {};
  params[0].operation          = hipStreamMemOpWriteValue32;
  params[0].writeValue.address = devPtr;
  params[0].writeValue.value   = 1000;
  params[0].writeValue.flags   = hipStreamWriteValueDefault;

  params[1].operation          = hipStreamMemOpWaitValue32;
  params[1].waitValue.address  = devPtr;
  params[1].waitValue.value    = 1000;
  params[1].waitValue.flags    = hipStreamWaitValueEq;

  hipBatchMemOpNodeParams nodeParams = {};
  nodeParams.ctx        = ctx;
  nodeParams.count      = 2;
  nodeParams.paramArray = params;
  nodeParams.flags      = 0;

  // Build graph with single BatchMemOp node
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &nodeParams));

  // Instantiate — AQL packet for batchMemOp blit kernel pre-built here
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Launch and verify
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t result = 0;
  HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  REQUIRE(result == 1000);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
#if HT_AMD
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
#endif
  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
