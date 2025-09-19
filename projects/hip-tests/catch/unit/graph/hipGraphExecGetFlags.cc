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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
/**
 * @addtogroup hipGraphExecGetFlags hipGraphExecGetFlags
 * @{
 * @ingroup GraphTest
 * `hipGraphExecGetFlags(hipGraphExec_t  graphExec,
 * unsigned long long *flags)` -
 * Return the flags on executable graph
 */
/**
 * Test Description
 * ------------------------
 *    - Verify API behavior with invalid arguments:
 *      -# graphExec is nullptr
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraphExecGetFlags_Negative") {
  hipGraphExec_t graphExec;
  unsigned long long flags;  // NOLINT
  constexpr size_t Nbytes = 10 * sizeof(int);

  hipGraphNode_t allocNodeA;
  hipMemAllocNodeParams allocParam;

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam));
  REQUIRE(allocParam.dptr != nullptr);

  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));
  HIP_CHECK_ERROR(hipGraphExecGetFlags(nullptr, &flags), hipErrorInvalidValue);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - This test will verify the flags what we set while initiating
 *      graph will be matching with flags values getting from
 *      the hipGraphExecGetFlags API call.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraphExecGetFlags_positive") {
  hipGraphExec_t graphExec;
  unsigned long long flags;  // NOLINT
  hipGraph_t graph;
  constexpr size_t Nbytes = 10 * sizeof(int);

  hipGraphNode_t allocNodeA;
  hipMemAllocNodeParams allocParam;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam));
  REQUIRE(allocParam.dptr != nullptr);

  SECTION("flag is 0") {
    HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));

    HIP_CHECK(hipGraphExecGetFlags(graphExec, &flags));
    REQUIRE(flags == 0);
  }
  SECTION("flag is hipGraphInstantiateFlagAutoFreeOnLaunch") {
    HIP_CHECK(
        hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

    HIP_CHECK(hipGraphExecGetFlags(graphExec, &flags));
    REQUIRE(flags == hipGraphInstantiateFlagAutoFreeOnLaunch);
  }

// The below feature flags are not implemented
// hipGraphInstantiateFlagUpload
// hipGraphInstantiateFlagDeviceLaunch
// hipGraphInstantiateFlagUseNodePriority
#if 0
  SECTION("flag is hipGraphInstantiateFlagUpload") {
    HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph,
                  hipGraphInstantiateFlagUpload));

    HIP_CHECK(hipGraphExecGetFlags(graphExec, &flags));
    REQUIRE(flags == hipGraphInstantiateFlagUpload);
  }

  SECTION("flag is hipGraphInstantiateFlagDeviceLaunch") {
    HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph,
                  hipGraphInstantiateFlagDeviceLaunch));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    REQUIRE(flags == hipGraphInstantiateFlagDeviceLaunch);
  }
  SECTION("flag is hipGraphInstantiateFlagUseNodePriority") {
    HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph,
                  hipGraphInstantiateFlagUseNodePriority));
    HIP_CHECK(hipGraphExecGetFlags(graphExec, &flags));
    REQUIRE(flags == hipGraphInstantiateFlagUseNodePriority);
  }
#endif
  HIP_CHECK(hipGraphDestroy(graph));
}
/**
 * End doxygen group GraphTest.
 * @}
 */
