/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "vulkan_test.hh"
#include "signal_semaphore_common.hh"

/**
 * @addtogroup hipGraphExternalSemaphoresSignalNodeSetParams
 * hipGraphExternalSemaphoresSignalNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphExternalSemaphoresSignalNodeSetParams(hipGraphNode_t hNode, const
 * hipExternalSemaphoreSignalNodeParams* nodeParams)` - Updates node parameters in the external
 * semaphore signal node.
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values. The graph is run and it is
 * verified that the graph node signals the external binary semaphore and operation finishes
 * successfully.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipGraphExternalSemaphoresSignalNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphExternalSemaphoresSignalNodeSetParams_Positive_Basic") {
  SignalExternalSemaphoreCommon(GraphExtSemaphoreSignalWrapper<true>);
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values. The graph is run and it is
 * verified that the graph node signals the external timeline semaphore and operation finishes
 * successfully.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipGraphExternalSemaphoresSignalNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphExternalSemaphoresSignalNodeSetParams_Vulkan_Positive_Timeline_Semaphore") {
  SignalExternalTimelineSemaphoreCommon(GraphExtSemaphoreSignalWrapper<true>);
}

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values. The graph is run and it is
 * verified that the graph node signals the external binary semaphores and operation finishes
 * successfully.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipGraphExternalSemaphoresSignalNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE(
    "Unit_hipGraphExternalSemaphoresSignalNodeSetParams_Vulkan_Positive_Multiple_Semaphores") {
  SignalExternalMultipleSemaphoresCommon(GraphExtSemaphoreSignalWrapper<true>);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphExternalSemaphoresSignalNodeSetParams behavior with invalid
 * arguments:
 *    -# Nullptr graph node
 *    -# Nullptr params
 * Test source
 * ------------------------
 *  - /unit/vulkan_interop/hipGraphExternalSemaphoresSignalNodeSetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphExternalSemaphoresSignalNodeSetParams_Vulkan_Negative_Parameters") {
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = 1;
  auto hip_ext_semaphore = ImportBinarySemaphore(vkt);

  hipExternalSemaphoreSignalNodeParams node_params = {};
  node_params.extSemArray = &hip_ext_semaphore;
  node_params.paramsArray = &signal_params;
  node_params.numExtSems = 1;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresSignalNodeSetParams(nullptr, &node_params),
                    hipErrorInvalidValue);
  }

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddExternalSemaphoresSignalNode(&node, graph, nullptr, 0, &node_params));

  SECTION("params == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresSignalNodeSetParams(node, nullptr),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
