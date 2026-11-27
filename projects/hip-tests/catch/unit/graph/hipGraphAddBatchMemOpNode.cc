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
 * - Verify the Negative cases of the hipGraphAddBatchMemOpNode API.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddBatchMemOpNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipGraphAddBatchMemOpNode_NegativeTsts) {
  HIP_CHECK(hipInit(0));
  hipGraph_t graph;
  hipCtx_t ctx;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));
  // Create a HIP graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  INFO("Graph created.");
  static hipStreamBatchMemOpParams paramArray[2];
  std::vector<hipDeviceptr_t> opsArray(1);
  HIP_CHECK(hipMalloc((void**)&opsArray[0], sizeof(uint32_t)));

  paramArray[0].operation = hipStreamMemOpWriteValue32;
  paramArray[0].writeValue.address = opsArray[0];
  paramArray[0].writeValue.value = 1000;
  paramArray[0].writeValue.flags = 0x0;
  paramArray[0].writeValue.alias = 0;

  paramArray[1].operation = hipStreamMemOpWaitValue32;
  paramArray[1].waitValue.address = opsArray[0];
  paramArray[1].waitValue.value = 1000;
  paramArray[1].waitValue.flags = hipStreamWaitValueEq;

  int totalOps = 2;
  // Setup the batch memory operation node parameters
  hipBatchMemOpNodeParams batchNodeParams, inValidBatchNodeParams;
  batchNodeParams.ctx = ctx;                // Use the current HIP context
  batchNodeParams.count = totalOps;         // Total number of memory operations
  batchNodeParams.paramArray = paramArray;  // Pointer to the array of memory operations
  batchNodeParams.flags = 0;                // No special flags

  inValidBatchNodeParams.ctx = ctx;
  inValidBatchNodeParams.count = -2;
  inValidBatchNodeParams.paramArray = nullptr;
  inValidBatchNodeParams.flags = -2;

  hipGraphNode_t batchMemOpNode;

  SECTION("Graph as null pointer") {
    HIP_CHECK_ERROR(
        hipGraphAddBatchMemOpNode(&batchMemOpNode, nullptr, nullptr, 0, &batchNodeParams),
        hipErrorInvalidValue);
  }

  SECTION("Number of Dependencies as Negative value") {
    HIP_CHECK_ERROR(
        hipGraphAddBatchMemOpNode(&batchMemOpNode, graph, nullptr, -2, &batchNodeParams),
        hipErrorInvalidValue);
  }
// Disabled for NVIDIA due to the defect SWDEV-502247
#if HT_AMD
  SECTION("Node Params as nullptr") {
    HIP_CHECK_ERROR(hipGraphAddBatchMemOpNode(&batchMemOpNode, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("Graph Node as nullptr") {
    HIP_CHECK_ERROR(hipGraphAddBatchMemOpNode(nullptr, graph, nullptr, 0, &batchNodeParams),
                    hipErrorInvalidValue);
  }
// Disabled due to defect SWDEV-502219
#if 0
  SECTION("Invalid node params") {
    HIP_CHECK_ERROR(hipGraphAddBatchMemOpNode(&batchMemOpNode, graph, nullptr,
                                              0, &inValidBatchNodeParams),
                    hipErrorInvalidValue);
  }
#endif
  HIP_CHECK(hipFree((void*)opsArray[0]));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipCtxDestroy(ctx));
}
namespace {
// Skips unless batch memory operations can run; returns signal memory for a 32-bit write.
bool SetUpBatchWrite(hipCtx_t* ctx, hipDeviceptr_t* devPtr) {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  int waitValueSupport = 0;
  if (hipDeviceGetAttribute(&waitValueSupport, hipDeviceAttributeCanUseStreamWaitValue, 0) !=
          hipSuccess ||
      waitValueSupport == 0) {
    return false;
  }
#if !HT_AMD
  return false;
#else
  HIP_CHECK(hipCtxCreate(ctx, 0, device));
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(devPtr), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  HIP_CHECK(hipMemset(reinterpret_cast<void*>(*devPtr), 0, sizeof(uint64_t)));
  return true;
#endif
}

hipBatchMemOpNodeParams WriteParams(hipCtx_t ctx, hipStreamBatchMemOpParams* op,
                                    hipDeviceptr_t devPtr, uint32_t value) {
  *op = {};
  op->operation = hipStreamMemOpWriteValue32;
  op->writeValue.address = devPtr;
  op->writeValue.value = value;
  op->writeValue.flags = hipStreamWriteValueDefault;
  hipBatchMemOpNodeParams nodeParams = {};
  nodeParams.ctx = ctx;
  nodeParams.count = 1;
  nodeParams.paramArray = op;
  return nodeParams;
}

uint32_t LaunchAndRead(hipGraphExec_t graphExec, hipDeviceptr_t devPtr) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  uint32_t result = 0;
  HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(result),
                      hipMemcpyDeviceToHost));
  return result;
}
}  // namespace

/**
 * Test Description
 * ------------------------
 * - The node keeps its own copy of the operation array: changing the caller's array after
 *   hipGraphAddBatchMemOpNode does not change what the graph writes.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddBatchMemOpNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipGraphAddBatchMemOpNode_ParamArrayCopied) {
  hipCtx_t ctx;
  hipDeviceptr_t devPtr = 0;
  if (!SetUpBatchWrite(&ctx, &devPtr)) {
    HIP_SKIP_TEST("Batch memory operations are not supported on this device.");
  }
  hipStreamBatchMemOpParams op;
  const hipBatchMemOpNodeParams nodeParams = WriteParams(ctx, &op, devPtr, 1000);

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &nodeParams));
  op.writeValue.value = 2000;  // the caller reuses its array
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(LaunchAndRead(graphExec, devPtr) == 1000);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 * - hipGraphExecUpdate from a graph whose batch memory operation node writes another value
 *   takes effect on the next launch.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddBatchMemOpNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_BatchMemOpNode) {
  hipCtx_t ctx;
  hipDeviceptr_t devPtr = 0;
  if (!SetUpBatchWrite(&ctx, &devPtr)) {
    HIP_SKIP_TEST("Batch memory operations are not supported on this device.");
  }
  hipStreamBatchMemOpParams ops[2];
  hipGraph_t graphs[2];
  const uint32_t values[2] = {1000, 2000};
  for (int i = 0; i < 2; ++i) {
    const hipBatchMemOpNodeParams nodeParams = WriteParams(ctx, &ops[i], devPtr, values[i]);
    HIP_CHECK(hipGraphCreate(&graphs[i], 0));
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddBatchMemOpNode(&node, graphs[i], nullptr, 0, &nodeParams));
  }
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graphs[0], nullptr, nullptr, 0));
  REQUIRE(LaunchAndRead(graphExec, devPtr) == 1000);

  hipGraphNode_t errorNode = nullptr;
  hipGraphExecUpdateResult updateResult = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(graphExec, graphs[1], &errorNode, &updateResult));
  REQUIRE(updateResult == hipGraphExecUpdateSuccess);
  REQUIRE(LaunchAndRead(graphExec, devPtr) == 2000);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graphs[1]));
  HIP_CHECK(hipGraphDestroy(graphs[0]));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
