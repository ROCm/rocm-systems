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
TEST_CASE("Unit_hipGraphAddBatchMemOpNode_NegativeTsts") {
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
/**
 * End doxygen group GraphTest.
 * @}
 */
