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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * @addtogroup hipGraphKernelNodeSetAttribute hipGraphKernelNodeSetAttribute
 * @{
 * @ingroup GraphTest
 * `hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode,
 *          hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* value )` -
 * Sets node attribute.
 */

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hipGraphKernelNodeSetAttribute
 *    1) Check hipGraphKernelNodeSetAttribute for AccessPolicyWindow attributes
 *    2) Check hipGraphKernelNodeSetAttribute for cooperative attributes
 *    3) Check hipGraphKernelNodeSetAttribute for window cooperative attributes
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphKernelNodeGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

static bool validateKernelNodeAttrValue(hipKernelNodeAttrValue in, hipKernelNodeAttrValue out) {
  if ((in.accessPolicyWindow.base_ptr != out.accessPolicyWindow.base_ptr) ||
      (in.accessPolicyWindow.hitProp != out.accessPolicyWindow.hitProp) ||
      (in.accessPolicyWindow.hitRatio != out.accessPolicyWindow.hitRatio) ||
      (in.accessPolicyWindow.missProp != out.accessPolicyWindow.missProp) ||
      (in.accessPolicyWindow.num_bytes != out.accessPolicyWindow.num_bytes) ||
      (in.cooperative != out.cooperative)) {
    return false;
  }
  return true;
}

TEST_CASE("Unit_hipGraphKernelNodeSetAttribute_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kernel_vecAdd;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpy_C, 1));

  hipKernelNodeAttrValue value_in, value_out;

  SECTION("Check hipGraphKernelNodeSetAttribute for AccessPolicyWindow") {
    memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
    memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));

    value_in.accessPolicyWindow.hitRatio = 0.8;
    value_in.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
    value_in.accessPolicyWindow.missProp = hipAccessPropertyStreaming;

    HIP_CHECK(hipGraphKernelNodeSetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_out));
    REQUIRE(true == validateKernelNodeAttrValue(value_in, value_out));
  }
  SECTION("Check hipGraphKernelNodeSetAttribute for cooperative") {
    memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
    memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));

    value_in.cooperative = 2;

    HIP_CHECK(hipGraphKernelNodeSetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_out));
    REQUIRE(true == validateKernelNodeAttrValue(value_in, value_out));
  }

  SECTION("Check hipGraphKernelNodeSetAttribute for window and cooperative") {
    memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
    memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));

    value_in.cooperative = 8;
    value_in.accessPolicyWindow.hitRatio = 0.1;
    value_in.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
    value_in.accessPolicyWindow.missProp = hipAccessPropertyNormal;

    HIP_CHECK(hipGraphKernelNodeSetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_out));
    REQUIRE(true == validateKernelNodeAttrValue(value_in, value_out));
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Negative/argument Test for API - hipGraphKernelNodeSetAttribute
 *    1) Pass kernel node as nullptr for Set attribute api and verify
 *    2) Pass KernelNodeAttrID as invalid value for Set attribute api and verify
 *    3) Pass KernelNodeAttrID as INT_MAX value for Get attribute api and verify
 *    4) Pass KernelNodeAttrValue as nullptr for Set attribute api and verify
 *    5) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value missProp as hipAccessPropertyPersisting
 *    6) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value hitProp as hipAccessPropertyPersisting
 *    7) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value accessPolicyWindow.hitRatio as 1.4
 *    8) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value accessPolicyWindow.hitRatio as 0
 *    9) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value accessPolicyWindow.hitRatio as 1
 *    10) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value accessPolicyWindow.hitRatio as -1.8
 *    11) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value accessPolicyWindow.hitRatio as -0.6
 *    12) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass accessPolicyWindow.num_bytes as 1024 & hitRatio as 0.6
 *    13) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
 *            and pass accessPolicyWindow.num_bytes as 1 GB & hitRatio as -0.6
 *    14) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value accessPolicyWindow.num_bytes as 1 MB
 *    15) Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow
 *            and pass value base_ptr as nullptr
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphKernelNodeSetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipGraphKernelNodeSetAttribute_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kernel_vecAdd;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  hipError_t ret;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kNodeParams));

  hipKernelNodeAttrValue value_in, value_out;
  memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
  memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));
  HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                           &value_in));
  memcpy(&value_out, &value_in, sizeof(hipKernelNodeAttrValue));

  SECTION("Pass kernel node as nullptr for Set attribute api") {
    ret = hipGraphKernelNodeSetAttribute(nullptr, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass KernelNodeAttrID as invalid value for Set attribute api") {
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttrID(-1), &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass KernelNodeAttrID as INT_MAX value for Set attribute api") {
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttrID(INT_MAX), &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#if HT_AMD  // getting SIGSEGV error in Cuda Setup
  SECTION("Pass KernelNodeAttrValue as nullptr for Set attribute api") {
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#endif
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value missProp as hipAccessPropertyPersisting") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.missProp = hipAccessPropertyPersisting;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value hitProp as hipAccessPropertyPersisting") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipSuccess == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value accessPolicyWindow.hitRatio as 1.4") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.hitRatio = 1.4;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value accessPolicyWindow.hitRatio as 0") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.hitRatio = 0;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipSuccess == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value accessPolicyWindow.hitRatio as 1") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.hitRatio = 1;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipSuccess == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value accessPolicyWindow.hitRatio as -1.8") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.hitRatio = -1.8;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value accessPolicyWindow.hitRatio as -0.6") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.hitRatio = -0.6;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " & pass accessPolicyWindow.num_bytes as 1024 & hitRatio as 0.6") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.num_bytes = 1024;
    value_in.accessPolicyWindow.hitRatio = 0.6;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " & pass accessPolicyWindow.num_bytes as 1 GB & hitRatio as -0.6") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.num_bytes = 1024 * 1024 * 1024;
    value_in.accessPolicyWindow.hitRatio = -0.6;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value accessPolicyWindow.num_bytes as 1 MB") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.num_bytes = 1024 * 1024;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION(
      "Pass KernelNodeAttrID as hipKernelNodeAttributeAccessPolicyWindow"
      " and pass value base_ptr as nullptr") {
    memcpy(&value_in, &value_out, sizeof(hipKernelNodeAttrValue));
    value_in.accessPolicyWindow.base_ptr = nullptr;
    ret = hipGraphKernelNodeSetAttribute(kernel_vecAdd, hipKernelNodeAttributeAccessPolicyWindow,
                                         &value_in);
    REQUIRE(hipSuccess == ret);
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
