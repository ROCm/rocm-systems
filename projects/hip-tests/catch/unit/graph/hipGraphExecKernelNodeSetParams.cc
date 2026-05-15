/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
Test Case Scenarios :
Negative -
1) Pass hGraphExec as nullptr and verify api returns error code.
2) Pass node as nullptr and verify api returns error code.
3) Pass NodeParams as un-initialized structure object and verify api returns error code.
4) Pass pNodeParams as nullptr and verify api returns error code.
5) Pass NodeParams:func data member as nullptr and verify api returns error code.
Functional -
1) Instantiate a graph with kernel node, obtain executable graph and update
   the kernel node params with set and check it is taking effect.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Negative Test for API hipGraphExecKernelNodeSetParams
 */
TEST_CASE("Unit_hipGraphExecKernelNodeSetParams_Negative") {
  constexpr size_t N = 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraph_t graph;
  hipGraphNode_t kNode{};
  hipKernelNodeParams kNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));

  hipGraphNode_t empty_node;
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, &kNode, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Pass hipGraphExec as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(nullptr, kNode, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass Node as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, nullptr, &kNodeParams),
                    hipErrorInvalidValue);
  }

#if HT_AMD
  /* NodeParams null check is disabled on Nvidia as
   * this call gives SIGSEGV error in CUDA setup */
  SECTION("Pass NodeParams as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, nullptr),
                    hipErrorInvalidValue);
  }
#endif

#if HT_NVIDIA  // on AMD this returns hipErrorInvalidValue
  SECTION("Pass NodeParams func data member as nullptr") {
    kNodeParams.func = nullptr;
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams),
                    hipErrorInvalidDeviceFunction);
  }
#endif

#if HT_NVIDIA  // segfaults on AMD
  SECTION("Pass kernelParams data member as nullptr") {
    kNodeParams.kernelParams = nullptr;
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams),
                    hipErrorInvalidValue);
  }
#endif

#if HT_NVIDIA  // segfaults on AMD
  SECTION("node is not a kernel node") {
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, empty_node, &kNodeParams),
                    hipErrorInvalidValue);
  }
#endif

  SECTION("node is not instantiated") {
    HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));
    HIP_CHECK_ERROR(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams),
                    hipErrorInvalidValue);
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Functional Test for API Exec Kernel Params
 */
TEST_CASE("Unit_hipGraphExecKernelNodeSetParams_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode;
  hipKernelNodeParams kNodeParams{}, kNodeParams1{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  HIP_CHECK(
      hipGraphAddKernelNode(&kNode, graph, dependencies.data(), dependencies.size(), &kNodeParams));

  kNodeParams1.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kNodeParams1.gridDim = dim3(blocks);
  kNodeParams1.blockDim = dim3(threadsPerBlock);
  kNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs);

  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(), dependencies.size(),
                                    C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams1));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorSUB<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
