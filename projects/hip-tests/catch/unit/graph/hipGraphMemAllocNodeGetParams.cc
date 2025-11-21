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

/**
 * @addtogroup hipGraphMemAllocNodeGetParams hipGraphMemAllocNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* params_out)`
 *  Returns a memory alloc node's parameters.
 * `hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dptr_out)` -
 *  Returns a memory free node's parameters.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hipGraphMemAllocNodeGetParams
 *    Create a graph and add a node with hipGraphAddMemAllocNode
 *    and hipGraphAddMemFreeNode and launch it.
 *  1) Get alloc node by calling hipGraphMemAllocNodeGetParams and Validate.
 *  2) Get Free Node ptr by calling hipGraphMemFreeNodeGetParams and Validate.
 *  3) Check for multiple devices case.
 *  4) Allocate multiple alloc node and validate by calling its get param.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphMemAllocNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

static bool validateAllocParam(hipMemAllocNodeParams in, hipMemAllocNodeParams out,
                               bool accessDesc = false) {
  if (in.bytesize != out.bytesize) return false;
  if (in.poolProps.allocType != out.poolProps.allocType) return false;
  if (in.poolProps.location.id != out.poolProps.location.id) return false;
  if (in.poolProps.location.type != out.poolProps.location.type) return false;

  if (accessDesc) {
    if (in.accessDescs->location.type != out.accessDescs->location.type) {
      return false;
    }
    if (in.accessDescs->location.id != out.accessDescs->location.id) {
      return false;
    }
    if (in.accessDescs->flags != out.accessDescs->flags) {
      return false;
    }
  }

  return true;
}

static void hipGraphMemAllocNodeGetParams_Functional(unsigned deviceId = 0) {
  constexpr size_t N = 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA;
  hipMemAllocNodeParams params_in, params_out;

  HIP_CHECK(hipSetDevice(deviceId));
  HIP_CHECK(hipDeviceGraphMemTrim(deviceId));

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&params_in, 0, sizeof(hipMemAllocNodeParams));
  params_in.bytesize = Nbytes;
  params_in.poolProps.allocType = hipMemAllocationTypePinned;
  params_in.poolProps.location.id = deviceId;
  params_in.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &params_in));
  int* A_d = reinterpret_cast<int*>(params_in.dptr);
  REQUIRE(A_d != nullptr);

  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1, A_d));

  HIP_CHECK(hipGraphMemAllocNodeGetParams(allocNodeA, &params_out));
  // validate params_out with params_in
  REQUIRE(true == validateAllocParam(params_in, params_out));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipDeviceGraphMemTrim(deviceId));
}

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hipGraphMemAllocNodeGetParams
 *    Create a graph and add a node with hipGraphAddMemAllocNode
 *    and hipGraphAddMemFreeNode and launch it.
 *  1) Get alloc node by calling hipGraphMemAllocNodeGetParams and Validate it.
 *  2) Get Free node ptr by calling hipGraphMemFreeNodeGetParams and Validate it.
 *  3) Check for multiple devices case.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphMemAllocNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphMem_Alloc_Free_NodeGetParams_Functional") {
  hipGraphMemAllocNodeGetParams_Functional();
}

TEST_CASE("Unit_hipGraphMem_Alloc_Free_NodeGetParams_Functional_MultiDevice") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    for (int i = 0; i < numDevices; ++i) {
      hipGraphMemAllocNodeGetParams_Functional(i);
    }
  } else {
    SUCCEED("Skipped the testcase as there is no device to test.");
  }
}

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hipGraphMemAllocNodeGetParams
 *    Create a graph and add multiple node with hipGraphAddMemAllocNode
 *    and hipGraphAddMemFreeNode and launch it.
 *  1) Allocate multiple alloc node and validate by calling its get param.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphMemAllocNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphMem_Alloc_Free_NodeGetParams_Functional_2") {
  constexpr size_t N = 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  size_t NElem{N};

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipGraphNode_t allocNodeA, freeNodeA, allocNodeB, freeNodeB;
  hipGraphNode_t allocNodeC, freeNodeC;
  hipMemAllocNodeParams params_in, params_out;

  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays<int>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&params_in, 0, sizeof(params_in));
  params_in.bytesize = Nbytes;
  params_in.poolProps.allocType = hipMemAllocationTypePinned;
  params_in.poolProps.location.id = 0;
  params_in.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &params_in));
  REQUIRE(params_in.dptr != nullptr);
  A_d = reinterpret_cast<int*>(params_in.dptr);
  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeB, graph, &allocNodeA, 1, &params_in));
  REQUIRE(params_in.dptr != nullptr);
  B_d = reinterpret_cast<int*>(params_in.dptr);
  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeC, graph, &allocNodeB, 1, &params_in));
  REQUIRE(params_in.dptr != nullptr);
  C_d = reinterpret_cast<int*>(params_in.dptr);

  // Check shows that A_d, B_d & C_d DON'T share any virtual address each other
  REQUIRE(A_d != B_d);
  REQUIRE(B_d != C_d);
  REQUIRE(A_d != C_d);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, &allocNodeC, 1, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, &allocNodeC, 1, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, &kernel_vecAdd, 1, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(
      hipGraphAddMemFreeNode(&freeNodeA, graph, &memcpyD2H_C, 1, reinterpret_cast<void*>(A_d)));
  HIP_CHECK(
      hipGraphAddMemFreeNode(&freeNodeB, graph, &memcpyD2H_C, 1, reinterpret_cast<void*>(B_d)));
  HIP_CHECK(
      hipGraphAddMemFreeNode(&freeNodeC, graph, &memcpyD2H_C, 1, reinterpret_cast<void*>(C_d)));

  HIP_CHECK(hipGraphMemAllocNodeGetParams(allocNodeA, &params_out));
  REQUIRE(true == validateAllocParam(params_in, params_out));
  HIP_CHECK(hipGraphMemAllocNodeGetParams(allocNodeB, &params_out));
  REQUIRE(true == validateAllocParam(params_in, params_out));
  HIP_CHECK(hipGraphMemAllocNodeGetParams(allocNodeC, &params_out));
  REQUIRE(true == validateAllocParam(params_in, params_out));

  int temp[] = {0};
  HIP_CHECK(hipGraphMemFreeNodeGetParams(freeNodeA, reinterpret_cast<void*>(temp)));
  HIP_CHECK(hipGraphMemFreeNodeGetParams(freeNodeB, reinterpret_cast<void*>(temp)));
  HIP_CHECK(hipGraphMemFreeNodeGetParams(freeNodeC, reinterpret_cast<void*>(temp)));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays<int>(nullptr, nullptr, nullptr, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipDeviceGraphMemTrim(0));
}

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hipGraphMemAllocNodeGetParams. Create a graph and add a node with
 * hipGraphAddMemAllocNode and hipGraphAddMemFreeNode and launch it. Check both pool props and
 * access descriptor.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphMemAllocNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphMem_Alloc_Free_NodeGetParams_Functional_3") {
  constexpr auto element_count{512 * 1024 * 1024};
  constexpr size_t num_bytes = element_count * sizeof(int);

  hipGraphExec_t graph_exec;
  hipGraph_t graph;

  LinearAllocGuard<int> A_h =
      LinearAllocGuard<int>(LinearAllocs::malloc, element_count * sizeof(int));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = 0;
  desc.flags = hipMemAccessFlagsProtReadWrite;

  hipGraphNode_t alloc_node;
  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = num_bytes;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;
  alloc_param.accessDescs = &desc;
  alloc_param.accessDescCount = 1;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param));
  REQUIRE(alloc_param.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(alloc_param.dptr);

  hipMemAllocNodeParams get_alloc_params;
  HIP_CHECK(hipGraphMemAllocNodeGetParams(alloc_node, &get_alloc_params));
  REQUIRE(validateAllocParam(alloc_param, get_alloc_params, true) == true);

  constexpr int fill_value = 11;
  hipGraphNode_t memset_node;
  hipMemsetParams memset_params{};
  memset(&memset_params, 0, sizeof(memset_params));
  memset_params.dst = reinterpret_cast<void*>(A_d);
  memset_params.value = fill_value;
  memset_params.pitch = 0;
  memset_params.elementSize = sizeof(int);
  memset_params.width = element_count;
  memset_params.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_node, graph, &alloc_node, 1, &memset_params));

  hipGraphNode_t memcpy_node;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, &memset_node, 1, A_h.host_ptr(), A_d,
                                    num_bytes, hipMemcpyDeviceToHost));

  hipGraphNode_t free_node;
  HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, &memcpy_node, 1, (void*)A_d));

  void* dptr_out;
  HIP_CHECK(hipGraphMemFreeNodeGetParams(free_node, &dptr_out));
  REQUIRE(A_d == static_cast<int*>(dptr_out));

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ArrayFindIfNot(A_h.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Negative Test for API - hipGraphMemAllocNodeGetParams
 *  1) Pass MemAllocNode as nullptr
 *  2) Pass MemAllocNode as empty node
 *  3) Pass params_out as nullptr
 *  4) Pass MemFreeNode inplace of MemAllocNode in 1st arguments
 *  - Negative Test for API - hipGraphMemFreeNodeGetParams
 *  1) Pass MemFreeNode as nullptr
 *  2) Pass MemFreeNode as empty node
 *  3) Pass free pointer as nullptr
 *  4) Pass free pointer as invalid pointer
 *  5) Pass MemAllocNode inplace of MemFreeNode in 1st arguments
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphMemAllocNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphMem_Alloc_Free_NodeGetParams_Negative") {
  hipError_t ret;
  constexpr size_t N = 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA;
  hipMemAllocNodeParams params_in, params_out;

  HIP_CHECK(hipDeviceGraphMemTrim(0));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&params_in, 0, sizeof(hipMemAllocNodeParams));
  params_in.bytesize = Nbytes;
  params_in.poolProps.allocType = hipMemAllocationTypePinned;
  params_in.poolProps.location.id = 0;
  params_in.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &params_in));
  int* A_d = reinterpret_cast<int*>(params_in.dptr);
  REQUIRE(A_d != nullptr);

  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1, A_d));

  SECTION("Pass MemAllocNode as nullptr") {
    ret = hipGraphMemAllocNodeGetParams(nullptr, &params_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass MemAllocNode as empty node") {
    hipGraphNode_t allocNode_empty{};
    ret = hipGraphMemAllocNodeGetParams(allocNode_empty, &params_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass params_out as nullptr") {
    ret = hipGraphMemAllocNodeGetParams(allocNodeA, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass MemFreeNode inplace of MemAllocNode in 1st arguments") {
    ret = hipGraphMemAllocNodeGetParams(freeNodeA, &params_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  int temp[] = {0};
  SECTION("Pass MemFreeNode as nullptr") {
    ret = hipGraphMemFreeNodeGetParams(nullptr, reinterpret_cast<void*>(temp));
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass MemFreeNode as empty node") {
    hipGraphNode_t freeNode_empty{};
    ret = hipGraphMemFreeNodeGetParams(freeNode_empty, reinterpret_cast<void*>(temp));
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass free pointer as nullptr") {
    ret = hipGraphMemFreeNodeGetParams(freeNodeA, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass MemAllocNode inplace of MemFreeNode in 1st arguments") {
    ret = hipGraphMemFreeNodeGetParams(allocNodeA, reinterpret_cast<void*>(temp));
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipDeviceGraphMemTrim(0));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
