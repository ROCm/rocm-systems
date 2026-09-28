/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipGraphExecBatchMemOpNodeSetParams
 hipGraphExecBatchMemOpNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec,
 hipGraphNode_t hNode, const hipBatchMemOpNodeParams* nodeParams)`
 * - Sets the parameters for a batch mem op node in the given graphExec
 */

/**
 * Test Description
 * ------------------------
 * - Verify the Negative cases of the hipGraphExecBatchMemOpNodeSetParams API.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecBatchMemOpNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipGraphExecBatchMemOpNodeSetParams_NegativeTsts) {
  HIP_CHECK(hipInit(0));
  hipGraph_t graph, graph1;
  hipGraphExec_t graphExec;
  hipCtx_t ctx;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));
  // Create a HIP graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  INFO("Graph created.");

  static hipStreamBatchMemOpParams paramArray[2], newParamArray[2];
  std::vector<hipDeviceptr_t> opsArray(1);
  HIP_CHECK(hipMalloc((void**)&opsArray[0], sizeof(uint32_t)));

  paramArray[0].operation = hipStreamMemOpWriteValue32;
  paramArray[0].writeValue.address = opsArray[0];
  paramArray[0].writeValue.value = 1000;
  paramArray[0].writeValue.flags = 0x0;
  // paramArray[0].writeValue.alias = 0;

  paramArray[1].operation = hipStreamMemOpWaitValue32;
  paramArray[1].waitValue.address = opsArray[0];
  paramArray[1].waitValue.value = 1000;
  paramArray[1].waitValue.flags = hipStreamWaitValueEq;
  // paramArray[i].waitValue.alias = 0;

  int totalOps = 2;
  // Setup the batch memory operation node parameters
  hipBatchMemOpNodeParams batchNodeParams;
  batchNodeParams.ctx = ctx;                // Use the current HIP context
  batchNodeParams.count = totalOps;         // Total number of memory operations
  batchNodeParams.paramArray = paramArray;  // Pointer to the array of memory operations
  batchNodeParams.flags = 0;                // No special flags

  // Add a batch memory operation node to the graph
  hipGraphNode_t batchMemOpNode, batchMemOpNode_1;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&batchMemOpNode, graph, nullptr, 0, &batchNodeParams));
  HIP_CHECK(hipGraphAddBatchMemOpNode(&batchMemOpNode_1, graph1, nullptr, 0, &batchNodeParams));
  INFO("hipGraphAddBatchMemOpNode added successfully.");

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  INFO("Graph instantiated.");
  for (int i = 0; i < totalOps; i++) {
    newParamArray[i] = paramArray[i];
  }
  newParamArray[0].writeValue.value = 2000;
  newParamArray[1].waitValue.flags = hipStreamWaitValueGte;

  hipBatchMemOpNodeParams newBatchNodeParams;
  newBatchNodeParams.ctx = ctx;
  newBatchNodeParams.count = totalOps;
  newBatchNodeParams.paramArray = newParamArray;
  newBatchNodeParams.flags = 0;

  hipBatchMemOpNodeParams invalidNewBatchNodeParams;
  invalidNewBatchNodeParams.ctx = ctx;
  invalidNewBatchNodeParams.count = 400;
  invalidNewBatchNodeParams.paramArray = newParamArray;
  invalidNewBatchNodeParams.flags = -4;

  SECTION("Graph Executable as nullptr") {
    HIP_CHECK_ERROR(
        hipGraphExecBatchMemOpNodeSetParams(nullptr, batchMemOpNode, &newBatchNodeParams),
        hipErrorInvalidValue);
  }

  SECTION("Batch Memory Node as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(graphExec, nullptr, &newBatchNodeParams),
                    hipErrorInvalidValue);
  }
// Disabled for NVIDIA due to the defect SWDEV-502247
#if HT_AMD
  SECTION("Batch node Parameters as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(graphExec, batchMemOpNode, nullptr),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("Irrelevant Batch Node") {
    HIP_CHECK_ERROR(
        hipGraphExecBatchMemOpNodeSetParams(graphExec, batchMemOpNode_1, &newBatchNodeParams),
        hipErrorInvalidValue);
  }
// Disabled due to defect SWDEV-502219
#if 0
  SECTION("Invalid Batch Node Params") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(
                        graphExec, batchMemOpNode, &invalidNewBatchNodeParams),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("Unchanged Batch node Parameters") {
    HIP_CHECK_ERROR(
        hipGraphExecBatchMemOpNodeSetParams(graphExec, batchMemOpNode, &batchNodeParams),
        hipSuccess);
  }
  HIP_CHECK(hipFree((void*)opsArray[0]));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipCtxDestroy(ctx));
}
__global__ void BatchMemOpTypeGateKernel(int* output) { output[0] = 1; }

/**
 * Test Description
 * ------------------------
 * - Verify that the batch-mem-op node APIs reject handles of other node types instead of
 *   reinterpreting them (the node-type gate), for the template, exec and query entry points.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecBatchMemOpNodeSetParams.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecBatchMemOpNodeSetParams_WrongNodeType) {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t ctx;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipDeviceptr_t opsAddress;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&opsAddress), sizeof(uint32_t)));
  int* kernelOutput = nullptr;
  HIP_CHECK(hipMalloc(&kernelOutput, sizeof(int)));

  hipStreamBatchMemOpParams paramArray[1] = {};
  paramArray[0].operation = hipStreamMemOpWriteValue32;
  paramArray[0].writeValue.address = opsAddress;
  paramArray[0].writeValue.value = 1000;
  paramArray[0].writeValue.flags = 0;
  hipBatchMemOpNodeParams batchNodeParams = {};
  batchNodeParams.ctx = ctx;
  batchNodeParams.count = 1;
  batchNodeParams.paramArray = paramArray;
  batchNodeParams.flags = 0;

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t batchNode, kernelNode;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&batchNode, graph, nullptr, 0, &batchNodeParams));
  void* args[] = {&kernelOutput};
  hipKernelNodeParams kernelParams{};
  kernelParams.func = reinterpret_cast<void*>(BatchMemOpTypeGateKernel);
  kernelParams.gridDim = dim3(1);
  kernelParams.blockDim = dim3(1);
  kernelParams.kernelParams = args;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams));
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Exec setter with a kernel node handle") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(graphExec, kernelNode, &batchNodeParams),
                    hipErrorInvalidValue);
  }
  SECTION("Template setter with a kernel node handle") {
    HIP_CHECK_ERROR(hipGraphBatchMemOpNodeSetParams(kernelNode, &batchNodeParams),
                    hipErrorInvalidValue);
  }
  SECTION("Getter with a kernel node handle") {
    hipBatchMemOpNodeParams out = {};
    HIP_CHECK_ERROR(hipGraphBatchMemOpNodeGetParams(kernelNode, &out), hipErrorInvalidValue);
  }
  SECTION("Batch node itself still accepted") {
    HIP_CHECK(hipGraphExecBatchMemOpNodeSetParams(graphExec, batchNode, &batchNodeParams));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(kernelOutput));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(opsAddress)));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 * - Verify that hipGraphExecBatchMemOpNodeSetParams takes effect on the next launch (the
 *   instantiated packets are refreshed, not replayed with the old op array).
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecBatchMemOpNodeSetParams.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecBatchMemOpNodeSetParams_UpdateTakesEffect) {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  int waitValueSupport = 0;
  auto attrErr =
      hipDeviceGetAttribute(&waitValueSupport, hipDeviceAttributeCanUseStreamWaitValue, 0);
  if (attrErr != hipSuccess || waitValueSupport == 0) {
    HIP_SKIP_TEST("hipStreamWaitValue is not supported on this device.");
  }
#if !HT_AMD
  HIP_SKIP_TEST("hipMallocSignalMemory is not supported on non-AMD backends.");
#endif
  hipCtx_t ctx;
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipDeviceptr_t devPtr = 0;
#if HT_AMD
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtr), sizeof(uint64_t),
                                  hipMallocSignalMemory));
#endif
  HIP_CHECK(hipMemset(reinterpret_cast<void*>(devPtr), 0, sizeof(uint64_t)));

  hipStreamBatchMemOpParams params[1] = {};
  params[0].operation = hipStreamMemOpWriteValue32;
  params[0].writeValue.address = devPtr;
  params[0].writeValue.value = 1000;
  params[0].writeValue.flags = hipStreamWriteValueDefault;
  hipBatchMemOpNodeParams nodeParams = {};
  nodeParams.ctx = ctx;
  nodeParams.count = 1;
  nodeParams.paramArray = params;
  nodeParams.flags = 0;

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &nodeParams));
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  auto launch_and_read = [&]() {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    uint32_t result = 0;
    HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    return result;
  };
  REQUIRE(launch_and_read() == 1000);

  params[0].writeValue.value = 2000;
  HIP_CHECK(hipGraphExecBatchMemOpNodeSetParams(graphExec, node, &nodeParams));
  REQUIRE(launch_and_read() == 2000);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
