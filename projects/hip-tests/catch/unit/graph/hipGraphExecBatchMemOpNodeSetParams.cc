/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
TEST_CASE("Unit_hipGraphExecBatchMemOpNodeSetParams_NegativeTsts") {
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
/**
 * End doxygen group GraphTest.
 * @}
 */
