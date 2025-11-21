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
 * @addtogroup hipGraphExternalSemaphoresSignalNodeGetParams
 * hipGraphExternalSemaphoresSignalNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphExternalSemaphoresSignalNodeGetParams(hipGraphNode_t hNode,
 * hipExternalSemaphoreSignalNodeParams* params_out)` - Returns external semaphore signal node
 * params.
 * ________________________
 * Test cases from other APIs:
 *  - @ref Unit_hipGraphExternalSemaphoresSignalNodeSetParams_Positive_Basic
 *  - @ref Unit_hipGraphExternalSemaphoresSignalNodeSetParams_Vulkan_Positive_Timeline_Semaphore
 *  - @ref Unit_hipGraphExternalSemaphoresSignalNodeSetParams_Vulkan_Positive_Multiple_Semaphores
 */

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphExternalSemaphoresSignalNodeGetParams behavior with invalid
 * arguments:
 *    -# Nullptr graph node
 *    -# Nullptr params
 *    -# Node is destroyed
 * Test source
 * ------------------------
 *  - /unit/vulkan_interop/hipGraphExternalSemaphoresSignalNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphExternalSemaphoresSignalNodeGetParams_Negative_Parameters") {
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
  hipExternalSemaphoreSignalNodeParams retrieved_params;

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddExternalSemaphoresSignalNode(&node, graph, nullptr, 0, &node_params));

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresSignalNodeGetParams(nullptr, &retrieved_params),
                    hipErrorInvalidValue);
  }

  SECTION("params_out == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresSignalNodeGetParams(node, nullptr),
                    hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-208
#if HT_NVIDIA
  SECTION("Node is destroyed") {
    hipGraph_t graph_temp = nullptr;
    HIP_CHECK(hipGraphCreate(&graph_temp, 0));
    hipGraphNode_t node_temp = nullptr;
    HIP_CHECK(
        hipGraphAddExternalSemaphoresSignalNode(&node_temp, graph_temp, nullptr, 0, &node_params));
    HIP_CHECK(hipGraphDestroy(graph_temp));
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresSignalNodeGetParams(node_temp, &retrieved_params),
                    hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group GraphTest.
 * @}
 */
