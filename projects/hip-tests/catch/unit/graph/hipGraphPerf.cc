/*
Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * @addtogroup hipGraphPerfCheck hipGraphPerfCheck
 * @{
 * @ingroup GraphTest
 * `hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
 *     const hipGraphNode_t* pDependencies, size_t numDependencies,
 *     const hipKernelNodeParams* pNodeParams)` -
 * Creates a kernel execution node and adds it to a graph.
 * Optimize HIPGraph Performance.
 */

#ifdef __linux__  // windows machine build failing refer ticket SWDEV-440611

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_defgroups.hh>

#ifdef _WIN64
#define setenv(x, y, z) _putenv_s(x, y)
#endif

static constexpr int N = 1024;
static constexpr int Nbytes = N * sizeof(int);
static size_t NElem{N};
static constexpr int blocksPerCU = 6;  // to hide latency
static constexpr int threadsPerBlock = 256;

__device__ int globalTo1[N];
__device__ int globalTo2[N];
__device__ int globalFrom1[N];
__device__ int globalFrom2[N];

static bool verifyVectorSquare(int* A_h, int* C_h, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      INFO("VectorSquare A and C not matching at " << i);
      return false;
    }
  }
  return true;
}

/*  - Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
      sequence and copy back the result and verify. */
static void checkGraphcontinuousKernelCall(const unsigned int kNumNode) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added multiple nodes of MemCpy, Kernel node continuously for
      2 block & copy back result in MemCpy. */
static void checkGraphcontinuousKernelCallIn2Blocks(const unsigned int kNumNode1,
                                                    const unsigned int kNumNode2) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4;
  std::vector<hipGraphNode_t> kNode1(kNumNode1), kNode2(kNumNode2);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode1; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode1[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode1[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode1[i - 1], &kNode1[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode1[kNumNode1 - 1], &memCpy3, 1));

  for (int i = 0; i < kNumNode2; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode2[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &kNode2[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode2[i - 1], &kNode2[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy4, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode2[kNumNode2 - 1], &memCpy4, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added 2 nodes of MemCpy, Kernel node whicl compute the operation & copy
      back result using MemCpy node. Call this multiple times sequentially. */
static void checkGraphMemcpyKernelMixCall(const unsigned int kNumIter) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  constexpr int kNumNode = 3;
  std::vector<hipGraphNode_t> node(kNumIter * kNumNode);
  std::vector<hipGraphNode_t> kNode(kNumIter);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int i = 0;
  for (int iter = 0; iter < kNumIter; iter++) {
    i = kNumNode * iter;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i], graph, nullptr, 0, A_d, A_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i + 1], graph, nullptr, 0, B_d, B_h, Nbytes,
                                      hipMemcpyHostToDevice));

    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[iter], graph, nullptr, 0, &kernelNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i + 2], graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    if (i != 0) HIP_CHECK(hipGraphAddDependencies(graph, &node[i - 1], &node[i], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i], &node[i + 1], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i + 1], &kNode[iter], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kNode[iter], &node[i + 2], 1));
  }

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added a nodes of MemCpy, MesSet, Kernel to do operation and copy back
      result using MemCpy node and call above operation in sequence. */
static void checkGraphMemcpyMemsetKernelMixCall(const unsigned int kNumIter) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  constexpr int kNumNode = 3;
  std::vector<hipGraphNode_t> node(kNumIter * kNumNode);
  std::vector<hipGraphNode_t> kNode(kNumIter);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int pitch_M = 0;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int i = 0;
  for (int iter = 0; iter < kNumIter; iter++) {
    i = kNumNode * iter;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i], graph, nullptr, 0, A_d, A_h, Nbytes,
                                      hipMemcpyHostToDevice));

    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(B_d);
    memsetParams.value = 2;
    memsetParams.pitch = pitch_M;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = N;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&node[i + 1], graph, nullptr, 0, &memsetParams));

    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[iter], graph, nullptr, 0, &kernelNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i + 2], graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    if (i != 0) HIP_CHECK(hipGraphAddDependencies(graph, &node[i - 1], &node[i], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i], &node[i + 1], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i + 1], &kNode[iter], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kNode[iter], &node[i + 2], 1));
  }

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  REQUIRE(true == verifyVectorSquare(A_h, C_h, N));

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added a EventRecordNode at start and 2 nodes of MemCpy, and multiple
      node of Kernel call in continuous sequence and copy back the result and
      add EventRecordNode at the end and verify the result. */
static void checkGraphEventcontinuousKernelCall(const unsigned int kNumNode) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t event_start, event_final;
  hipEvent_t eventstart, eventend;
  HIP_CHECK(hipEventCreate(&eventstart));
  HIP_CHECK(hipEventCreate(&eventend));

  HIP_CHECK(hipGraphAddEventRecordNode(&event_start, graph, nullptr, 0, eventstart));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_final, graph, nullptr, 0, eventend));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start, &memCpy1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &event_final, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipEventSynchronize(eventend));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipEventDestroy(eventstart));
  HIP_CHECK(hipEventDestroy(eventend));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added a EventRecordNode at start and Added multiple nodes of MemCpy, Kernel
      node continuously and added one more EventRecordNode in mid for synchronize
      and do similar MemCpy, Kernel node continuously in 2nd block & copy back
      result in MemCpy and add EventRecordNode at the end for synchronize. */
static void checkGraphEventcontinuousKernelCallIn2Blocks(const unsigned int kNumNode1,
                                                         const unsigned int kNumNode2) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4;
  std::vector<hipGraphNode_t> kNode1(kNumNode1), kNode2(kNumNode2);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t event_start, event_mid, event_final;
  hipEvent_t eventstart, eventmid, eventend;
  HIP_CHECK(hipEventCreate(&eventstart));
  HIP_CHECK(hipEventCreate(&eventmid));
  HIP_CHECK(hipEventCreate(&eventend));

  HIP_CHECK(hipGraphAddEventRecordNode(&event_start, graph, nullptr, 0, eventstart));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_mid, graph, nullptr, 0, eventmid));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_final, graph, nullptr, 0, eventend));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start, &memCpy1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode1; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode1[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode1[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode1[i - 1], &kNode1[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode1[kNumNode1 - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &event_mid, 1));

  for (int i = 0; i < kNumNode2; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode2[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &kNode2[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode2[i - 1], &kNode2[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy4, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode2[kNumNode2 - 1], &memCpy4, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy4, &event_final, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipEventSynchronize(eventend));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipEventDestroy(eventstart));
  HIP_CHECK(hipEventDestroy(eventmid));
  HIP_CHECK(hipEventDestroy(eventend));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Add multiple nodes of MemCpy, Kernel in sequence multiple times.
 *  2) Add multiple nodes of MemCpy, MesSet, Kernel in sequence.
 *  3) Add multiple nodes of MemCpy, Kernel node continuously & copy back result in MemCpy.
 *  4) Add multiple nodes of MemCpy, Kernel node continuously for 2 block & copy back result in
 MemCpy.
 *  5) Add a EventRecordNode at start and 2 nodes of MemCpy, and multiple
       node of Kernel call in continuous sequence and copy back the result and
       add EventRecordNode at the end and verify the result.
 *  6) Add a EventRecordNode at start and Added multiple nodes of MemCpy, Kernel
      node continuously and added one more EventRecordNode in mid for synchronize
      and do similar MemCpy, Kernel node continuously in 2nd block & copy back
      result in MemCpy and add EventRecordNode at the end for synchronize.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipGraph_PerfCheck_MemcpyKernelMixCall") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  constexpr int kNumIter1 = 25;
  constexpr int kNumIter2 = 30;
  constexpr int kNumKNode1 = 15;
  constexpr int kNumKNode2 = 45;

  checkGraphMemcpyKernelMixCall(kNumIter1);
  checkGraphMemcpyMemsetKernelMixCall(kNumIter2);
  checkGraphcontinuousKernelCall(kNumKNode1);
  checkGraphcontinuousKernelCallIn2Blocks(kNumKNode1, kNumKNode2);
  checkGraphEventcontinuousKernelCall(kNumIter1);
  checkGraphEventcontinuousKernelCallIn2Blocks(kNumKNode1, kNumKNode2);

  checkGraphMemcpyKernelMixCall(kNumIter2);
  checkGraphMemcpyMemsetKernelMixCall(kNumIter1);
  checkGraphcontinuousKernelCall(kNumKNode2);
  checkGraphcontinuousKernelCallIn2Blocks(kNumKNode2, kNumKNode1);
  checkGraphEventcontinuousKernelCall(kNumIter2);
  checkGraphEventcontinuousKernelCallIn2Blocks(kNumKNode2, kNumKNode1);
}

static void hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams(const hipStream_t& stream) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode;
  hipKernelNodeParams kNodeParams{}, kNodeParams1{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

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

  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(), dependencies.size(),
                                    C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  kNodeParams1.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kNodeParams1.gridDim = dim3(blocks);
  kNodeParams1.blockDim = dim3(threadsPerBlock);
  kNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs);

  // Instantiate again and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, kNode, &kNodeParams1));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, and a node of Kernel call and
       Instantiate graph and update kernelNodeParams for last kernel
       and copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as used created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams(stream);
      }
    }
  }
}

#if HT_NVIDIA
static void hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams_inLoop(const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  hipKernelNodeParams kNodeParams1{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams1.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kNodeParams1.gridDim = dim3(blocks);
  kNodeParams1.blockDim = dim3(threadsPerBlock);
  kNodeParams1.sharedMemBytes = 0;
  kNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, kNode[kNumNode - 1], &kNodeParams1));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, & multiple node of Kernel call in continuous sequence
       and Instantiate graph & update kernelNodeParams with hipGraphExecKernelNodeSetParams
       for last kernel and copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams_inLoop") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as used created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams_inLoop(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams_inLoop(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecKernelNodeSetParams_inLoop(stream);
      }
    }
  }
}
#endif
/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 3 nodes of MemCpy, and node of Kernel call in continuous
      sequence and Instantiate graph and update hipGraphExecMemcpyNodeSetParams
      for source memCopy3 node and copy back the result and verify.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }
  constexpr int kNumNode = 1;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  int *A_d1, *B_d1, *C_d1, *A_h1, *B_h1, *C_h1;
  HipTest::initArrays(&A_d1, &B_d1, &C_d1, &A_h1, &B_h1, &C_h1, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNode[0], graph, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[0], 1));

  hipMemcpy3DParms myparams;
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(Nbytes, 1, 1);
  myparams.dstPtr = make_hipPitchedPtr(A_h1, Nbytes, Nbytes, 1);
  myparams.srcPtr = make_hipPitchedPtr(A_d1, Nbytes, Nbytes, 1);
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memCpy3, graph, nullptr, 0, &myparams));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Verifying with different memCopy node Params") {
    memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
    myparams.srcPos = make_hipPos(0, 0, 0);
    myparams.dstPos = make_hipPos(0, 0, 0);
    myparams.extent = make_hipExtent(Nbytes, 1, 1);
    myparams.dstPtr = make_hipPitchedPtr(C_h, Nbytes, Nbytes, 1);
    myparams.srcPtr = make_hipPitchedPtr(C_d, Nbytes, Nbytes, 1);
    myparams.kind = hipMemcpyDeviceToHost;

    HIP_CHECK(hipGraphExecMemcpyNodeSetParams(graphExec, memCpy3, &myparams));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HipTest::freeArrays(A_d1, B_d1, C_d1, A_h1, B_h1, C_h1, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

static void hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams_inLoop(const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int harray1D[N]{};
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  int *A_d1, *B_d1, *C_d1, *A_h1, *B_h1, *C_h1;
  HipTest::initArrays(&A_d1, &B_d1, &C_d1, &A_h1, &B_h1, &C_h1, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }

  hipMemcpy3DParms myparams;
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(Nbytes, 1, 1);
  myparams.dstPtr = make_hipPitchedPtr(harray1D, Nbytes, Nbytes, 1);
  myparams.srcPtr = make_hipPitchedPtr(C_d, Nbytes, Nbytes, 1);
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memCpy3, graph, nullptr, 0, &myparams));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, harray1D, N);

  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(Nbytes, 1, 1);
  myparams.dstPtr = make_hipPitchedPtr(C_h, Nbytes, Nbytes, 1);
  myparams.srcPtr = make_hipPitchedPtr(C_d, Nbytes, Nbytes, 1);
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphExecMemcpyNodeSetParams(graphExec, memCpy3, &myparams));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HipTest::freeArrays(A_d1, B_d1, C_d1, A_h1, B_h1, C_h1, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
      sequence and Instantiate graph and update hipGraphExecMemcpyNodeSetParams
      for source memCopy3 node and copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams_inLoop") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams_inLoop(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams_inLoop(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams_inLoop(stream);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams1D_inLoop(const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int* hData = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hData != nullptr);
  for (int i = 0; i < N; ++i) hData[i] = 2 * i + 1;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graphExec, memCpy2, B_d, hData, Nbytes,
                                              hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, hData, C_h, N);

  free(hData);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
      sequence and Instantiate graph and update hipGraphExecMemcpyNodeSetParams1D
      for source memCopy2 node and copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams1D_inLoop") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams1D_inLoop(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams1D_inLoop(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParams1D_inLoop(stream);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsFrmSymbol(const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memCpy3, graph, nullptr, 0, HIP_SYMBOL(globalFrom1), C_d,
                                          Nbytes, 0, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(
      &memCpy4, graph, nullptr, 0, C_h, HIP_SYMBOL(globalFrom2), Nbytes, 0, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &memCpy4, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol(
      graphExec, memCpy4, C_h, HIP_SYMBOL(globalFrom1), Nbytes, 0, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
      sequence and Instantiate graph and update hipGraphExecMemcpyNodeSetParamsFromSymbol
      for source memCopy4 node and copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsFrmSymbol") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsFrmSymbol(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsFrmSymbol(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsFrmSymbol(stream);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsToSymbol(const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memCpy3, graph, nullptr, 0, HIP_SYMBOL(globalTo1), C_d,
                                          Nbytes, 0, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memCpy4, graph, nullptr, 0, C_h, HIP_SYMBOL(globalTo2),
                                            Nbytes, 0, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &memCpy4, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec, memCpy3, HIP_SYMBOL(globalTo2), C_d,
                                                    Nbytes, 0, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
      sequence and Instantiate graph and update hipGraphExecMemcpyNodeSetParamsToSymbol
      for source memCopy3 node and copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsToSymbol") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsToSymbol(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsToSymbol(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecMemcpyNodeSetParamsToSymbol(stream);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams(const hipStream_t& stream,
                                                               int test) {
  constexpr int kNumNode = 35;
  constexpr int memSetVal = 7, memSetVal2 = 9;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memSet;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int size, elementSize;

  if (test == 0) {
    size = Nbytes;
    elementSize = sizeof(char);
  } else {
    size = N;
    elementSize = sizeof(int);
  }

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = A_d;
  memsetParams.value = memSetVal;
  memsetParams.elementSize = elementSize;
  memsetParams.width = size;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memSet, graph, nullptr, 0, &memsetParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &memSet, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memSet, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  if (test == 0) {
    memset(A_h, memSetVal, size);
  } else {
    for (int i = 0; i < N; i++) {
      A_h[i] = memSetVal;
    }
  }
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  // update MemSet Node using Exec
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = A_d;
  memsetParams.value = memSetVal2;
  memsetParams.elementSize = elementSize;
  memsetParams.width = size;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphExecMemsetNodeSetParams(graphExec, memSet, &memsetParams));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  if (test == 0) {
    memset(A_h, memSetVal2, size);
  } else {
    for (int i = 0; i < N; i++) {
      A_h[i] = memSetVal2;
    }
  }
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy and add a memSet node, and multiple node of Kernel
       call in continuous sequence and Instantiate graph and update memSet node with
       hipGraphExecMemsetNodeSetParams api and verify the result.
    i)   Verify the memset with reset 1 byte (char size) block.
    ii)  Verify the memset with reset 4 byte (int size) block.
    iii) Check with Multi device case.
    iv)  Pass stream as user created stream
    v)   Pass stream as default stream
    vi)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams(stream, 0);
        hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams(stream, 1);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams(stream, 0);
        hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams(stream, 1);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams(stream, 0);
        hipGraph_PerfCheck_hipGraphExecMemsetNodeSetParams(stream, 1);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams(const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  constexpr int memSetVal = 7, memSetVal2 = 9;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4;
  hipGraphNode_t memSet1, memSet2, childGraphNode;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph, childGraph1, childGraph2;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph1, 0));
  HIP_CHECK(hipGraphCreate(&childGraph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }

  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = C_d;
  memsetParams.value = memSetVal;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memSet1, childGraph1, nullptr, 0, &memsetParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, childGraph1, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &memSet1, &memCpy3, 1));
  // Adding childnode to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &childGraphNode, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  memset(A_h, memSetVal, Nbytes);
  for (unsigned int i = 0; i < N; i++) {
    if (A_h[i] != C_h[i]) {
      WARN("Validation failed at " << i << "\t" << C_h[i]);
      REQUIRE(false);
    }
  }

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = C_d;
  memsetParams.value = memSetVal2;
  memsetParams.elementSize = sizeof(int);
  memsetParams.width = N;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memSet2, childGraph2, nullptr, 0, &memsetParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy4, childGraph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &memSet2, &memCpy4, 1));

  // Update the childgraph node
  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode, childGraph2));
  // Launch Again and verify it once
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (unsigned int i = 0; i < N; i++) {
    if (memSetVal2 != C_h[i]) {
      WARN("Validation failed at " << i << "\t" << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(childGraph1));
  HIP_CHECK(hipGraphDestroy(childGraph2));
}

static void hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_Kernel(
    const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4, childGraphNode;
  hipGraphNode_t memCpyC11, memCpyC12, memCpyC13, kNodeC1;
  hipGraphNode_t memCpyC21, memCpyC22, memCpyC23, kNodeC2;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph, childGraph1, childGraph2;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  int *A_d1, *B_d1, *C_d1, *A_h1, *B_h1, *C_h1;
  HipTest::initArrays(&A_d1, &B_d1, &C_d1, &A_h1, &B_h1, &C_h1, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph1, 0));
  HIP_CHECK(hipGraphCreate(&childGraph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }

  // Add child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC11, childGraph1, nullptr, 0, A_d1, A_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC12, childGraph1, nullptr, 0, B_d1, B_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &memCpyC11, &memCpyC12, 1));

  hipKernelNodeParams kNodeParams{};
  void* kernelArgsC[] = {&A_d1, &C_d1, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsC);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeC1, childGraph1, nullptr, 0, &kNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC13, childGraph1, nullptr, 0, C_h1, C_d1, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &memCpyC12, &kNodeC1, 1));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &kNodeC1, &memCpyC13, 1));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy4, childGraph1, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &memCpyC13, &memCpy4, 1));

  // Adding childnode to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &childGraphNode, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);           // MainGraph o/p verification
  REQUIRE(true == verifyVectorSquare(A_h1, C_h1, N));  // ChildGraph o/p verify

  // new child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC21, childGraph2, nullptr, 0, A_d1, A_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC22, childGraph2, nullptr, 0, B_d1, B_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &memCpyC21, &memCpyC22, 1));

  memset(&kNodeParams, 0x00, sizeof(hipKernelNodeParams));
  void* kernelArgC[] = {&A_d1, &B_d1, &C_d1, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgC);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeC2, childGraph2, nullptr, 0, &kNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC23, childGraph2, nullptr, 0, C_h1, C_d1, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &memCpyC22, &kNodeC2, 1));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &kNodeC2, &memCpyC23, 1));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, childGraph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &memCpyC23, &memCpy3, 1));

  // Update the childgraph node
  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode, childGraph2));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify modified graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);     // MainGraph o/p verification
  HipTest::checkVectorADD(A_h1, B_h1, C_h1, N);  // ChildGraph o/p verification

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HipTest::freeArrays(A_d1, B_d1, C_d1, A_h1, B_h1, C_h1, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(childGraph1));
  HIP_CHECK(hipGraphDestroy(childGraph2));
}

static void hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_mKernel(
    const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  constexpr int kNumNodeChild = 45;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4, childGraphNode;
  hipGraphNode_t memCpyC11, memCpyC12, memCpyC13;
  hipGraphNode_t memCpyC21, memCpyC22, memCpyC23;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  std::vector<hipGraphNode_t> kNodeC1(kNumNodeChild);
  std::vector<hipGraphNode_t> kNodeC2(kNumNodeChild);
  hipGraph_t graph, childGraph1, childGraph2;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  int *A_d1, *B_d1, *C_d1, *A_h1, *B_h1, *C_h1;
  HipTest::initArrays(&A_d1, &B_d1, &C_d1, &A_h1, &B_h1, &C_h1, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph1, 0));
  HIP_CHECK(hipGraphCreate(&childGraph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }

  // Add child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC11, childGraph1, nullptr, 0, A_d1, A_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC12, childGraph1, nullptr, 0, B_d1, B_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &memCpyC11, &memCpyC12, 1));

  for (int i = 0; i < kNumNodeChild; i++) {
    hipKernelNodeParams kNodeParams{};
    void* kernelArgs[] = {&A_d1, &C_d1, reinterpret_cast<void*>(&NElem)};
    kNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
    kNodeParams.gridDim = dim3(blocks);
    kNodeParams.blockDim = dim3(threadsPerBlock);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNodeC1[i], childGraph1, nullptr, 0, &kNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(childGraph1, &memCpyC12, &kNodeC1[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(childGraph1, &kNodeC1[i - 1], &kNodeC1[i], 1));
    }
  }

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC13, childGraph1, nullptr, 0, C_h1, C_d1, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &kNodeC1[kNumNodeChild - 1], &memCpyC13, 1));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy4, childGraph1, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph1, &memCpyC13, &memCpy4, 1));

  // Adding childnode to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &childGraphNode, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);           // MainGraph o/p verification
  REQUIRE(true == verifyVectorSquare(A_h1, C_h1, N));  // ChildGraph o/p verify

  // new child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC21, childGraph2, nullptr, 0, A_d1, A_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC22, childGraph2, nullptr, 0, B_d1, B_h1, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &memCpyC21, &memCpyC22, 1));

  for (int i = 0; i < kNumNodeChild; i++) {
    hipKernelNodeParams kNodeParams{};
    void* kernelArgs[] = {&A_d1, &B_d1, &C_d1, reinterpret_cast<void*>(&NElem)};
    kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kNodeParams.gridDim = dim3(blocks);
    kNodeParams.blockDim = dim3(threadsPerBlock);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNodeC2[i], childGraph2, nullptr, 0, &kNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(childGraph2, &memCpyC22, &kNodeC2[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(childGraph2, &kNodeC2[i - 1], &kNodeC2[i], 1));
    }
  }

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpyC23, childGraph2, nullptr, 0, C_h1, C_d1, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &kNodeC2[kNumNodeChild - 1], &memCpyC23, 1));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, childGraph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph2, &memCpyC23, &memCpy3, 1));

  // Update the childgraph node
  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode, childGraph2));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify modified graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);     // MainGraph o/p verification
  HipTest::checkVectorADD(A_h1, B_h1, C_h1, N);  // ChildGraph o/p verification

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HipTest::freeArrays(A_d1, B_d1, C_d1, A_h1, B_h1, C_h1, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(childGraph1));
  HIP_CHECK(hipGraphDestroy(childGraph2));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
       sequence and add a child graph node in the end and Instantiate graph and
       update the graphExec with hipGraphExecChildGraphNodeSetParams for child
       graph node which will copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
    2) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
       sequence and add a child graph which contain a kernel operation and
       add this child graph as a node in the end of main graph & Instantiate graph.
       update the graphExec with hipGraphExecChildGraphNodeSetParams for child
       graph node with similar topology which will copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
    3) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
       sequence and add a child graph which contain a kernel call in continuous
       sequence operation and add this child graph as a node in the end of main
       graph & Instantiate main graph and launch and check result of main graph.
       update the graphExec with hipGraphExecChildGraphNodeSetParams for child
       graph node with similar topology which will copy back the result and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams(stream);
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_Kernel(stream);
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_mKernel(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams(stream);
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_Kernel(stream);
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_mKernel(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams(stream);
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_Kernel(stream);
        hipGraph_PerfCheck_hipGraphExecChildGraphNodeSetParams_mKernel(stream);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecEventRecordNodeSetEvent(const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipEvent_t event_start, event1_end, event2_end;
  HIP_CHECK(hipEventCreate(&event_start));
  HIP_CHECK(hipEventCreate(&event1_end));
  HIP_CHECK(hipEventCreate(&event2_end));

  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create nodes with event_start and event1_end
  hipGraphNode_t event_start_rec, event_end_rec;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_start_rec, graph, nullptr, 0, event_start));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start_rec, &memCpy1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_end_rec, graph, nullptr, 0, event1_end));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &event_end_rec, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);

  float t1 = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&t1, event_start, event1_end));
  REQUIRE(t1 > 0.0f);

  // Change the event at event_end_rec node to event2_end
  HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(graphExec, event_end_rec, event2_end));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);

  // Validate the changed events
  float t2 = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&t2, event_start, event2_end));
  REQUIRE(t2 > 0.0f);

  // Validate the changed events and initial event
  float t3 = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&t3, event1_end, event2_end));
  REQUIRE(t3 > 0.0f);

  // Free resources
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event_start));
  HIP_CHECK(hipEventDestroy(event1_end));
  HIP_CHECK(hipEventDestroy(event2_end));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added event start node at the begining and than add 2 nodes of MemCpy,
       and multiple node of Kernel call in continuous sequence and add a child
       graph node in the end and Instantiate graph and update the graphExec with
       hipGraphExecEventRecordNodeSetEvent and added a graph node which will copy
       back the result and add event end node at the end and verify the time elapse.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecEventRecordNodeSetEvent") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecEventRecordNodeSetEvent(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecEventRecordNodeSetEvent(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecEventRecordNodeSetEvent(stream);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent_waitKrnl(
    const hipStream_t& stream) {
  constexpr int kNumNode = 35;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  // Create events
  hipEvent_t event, event2;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t event_wait_node;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  hipGraphNode_t event_start_rec;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_start_rec, graph, nullptr, 0, event2));
  HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &event_start_rec, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start_rec, &event_wait_node, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecEventWaitNodeSetEvent(graphExec, event_wait_node, event2));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipEventDestroy(event2));
}

static void hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent(const hipStream_t& stream) {
  constexpr int kNumNode = 45;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  // Create events
  hipEvent_t event, event2;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t event_wait_node;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &event_wait_node, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecEventWaitNodeSetEvent(graphExec, event_wait_node, event2));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipEventDestroy(event2));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
       sequence and add a wait event node in the end and Instantiate graph and
       update the graphExec with hipGraphExecEventWaitNodeSetEvent node and a
       graph node which will copy back the result and add event end node at the
       end and verify.
    2) Added 2 nodes of MemCpy, and multiple node of Kernel call in continuous
       sequence and add a wait event node in the end and Instantiate graph and
       add a wait kernel and memcpy node to copy back the result.
       update the graphExec with hipGraphExecEventWaitNodeSetEvent node and a
       graph node which will copy back the result and add event end node at the
       end and verify.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as user created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent(stream);
        hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent_waitKrnl(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent(stream);
        hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent_waitKrnl(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent(stream);
        hipGraph_PerfCheck_hipGraphExecEventWaitNodeSetEvent_waitKrnl(stream);
      }
    }
  }
}

void callBackFunc_1(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < N; i++) {
    A[i] = i + i;
  }
}
static void callBackFunc_1_Verify(int* C_h) {
  for (int i = 0; i < N; i++) {
    if (C_h[i] != (i + i)) {
      INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }
}

void callBackFunc_2(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < N; i++) {
    A[i] = i * i;
  }
}
static void callBackFunc_2_Verify(int* C_h) {
  for (int i = 0; i < N; i++) {
    if (C_h[i] != (i * i)) {
      INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecHostNodeSetParams(const hipStream_t& stream) {
  constexpr int kNumNode = 45;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callBackFunc_1;
  hostParams.userData = C_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &hostNode, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify execution result
  callBackFunc_1_Verify(C_h);

  hipHostNodeParams sethostParam = {0, 0};
  sethostParam.fn = callBackFunc_2;
  sethostParam.userData = C_h;

  HIP_CHECK(hipGraphExecHostNodeSetParams(graphExec, hostNode, &sethostParam));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  callBackFunc_2_Verify(C_h);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy, & multiple node of Kernel call in continuous sequence
       and Instantiate graph & add a host node and launch the graph and verify result.
       Now update the host node parameters using api hipGraphExecHostNodeSetParams
       and verify the result which reflect modified data.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecHostNodeSetParams") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as used created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecHostNodeSetParams(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecHostNodeSetParams(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecHostNodeSetParams(stream);
      }
    }
  }
}

static void hipGraph_PerfCheck_hipGraphExecUpdate(const hipStream_t& stream) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  size_t NElem{N};
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipGraphNode_t memcpy_A2, memcpy_B2, memcpy_C2;
  hipGraphNode_t kernel_vecADD, kernel_vecSUB;
  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecADD, graph1, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph1, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_A, &kernel_vecADD, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_B, &kernel_vecADD, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernel_vecADD, &memcpy_C, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A2, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B2, graph2, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  memset(&kernelNodeParams, 0x00, sizeof(hipKernelNodeParams));
  void* kernelArgs1[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecSUB, graph2, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C2, graph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_A2, &kernel_vecSUB, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_B2, &kernel_vecSUB, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kernel_vecSUB, &memcpy_C2, 1));
  HIP_CHECK(hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}


/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy & a Kernel node and copy back result using memcpy
       and Instantiate graph & update new graph with similar node structure with
       api hipGraphExecUpdate and verify the result, the updated node should reflect.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecUpdate") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as used created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecUpdate(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecUpdate(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecUpdate(stream);
      }
    }
  }
}

#if HT_NVIDIA
static void hipGraph_PerfCheck_hipGraphExecUpdate_kernel_inLoop(const hipStream_t& stream) {
  constexpr int kNumNode = 45;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  hipGraphNode_t memCpy21, memCpy22, memCpy23;
  std::vector<hipGraphNode_t> kNode(kNumNode);
  std::vector<hipGraphNode_t> kNode2(kNumNode);
  hipGraph_t graph, graph2;
  hipGraphExec_t graphExec;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  int *A_d2, *B_d2, *C_d2, *A_h2, *B_h2, *C_h2;
  HipTest::initArrays(&A_d2, &B_d2, &C_d2, &A_h2, &B_h2, &C_h2, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i - 1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode - 1], &memCpy3, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy21, graph2, nullptr, 0, A_d2, A_h2, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy22, graph2, nullptr, 0, B_d2, B_h2, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memCpy21, &memCpy22, 1));

  for (int i = 0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParam{};
    void* kernelArgs2[] = {&A_d2, &B_d2, &C_d2, reinterpret_cast<void*>(&NElem)};
    kernelNodeParam.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kernelNodeParam.gridDim = dim3(blocks);
    kernelNodeParam.blockDim = dim3(threadsPerBlock);
    kernelNodeParam.sharedMemBytes = 0;
    kernelNodeParam.kernelParams = reinterpret_cast<void**>(kernelArgs2);
    kernelNodeParam.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode2[i], graph2, nullptr, 0, &kernelNodeParam));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph2, &memCpy22, &kNode2[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph2, &kNode2[i - 1], &kNode2[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy23, graph2, nullptr, 0, C_h2, C_d2, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kNode2[kNumNode - 1], &memCpy23, 1));

  HIP_CHECK(hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h2, B_h2, C_h2, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HipTest::freeArrays(A_d2, B_d2, C_d2, A_h2, B_h2, C_h2, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_PACKET_CAPTURE
 *  1) Added 2 nodes of MemCpy & a Kernel node in sequence and copy back result using memcpy
       and Instantiate graph & update new graph with similar node structure with
       api hipGraphExecUpdate and verify the result, the updated node should reflect.
    i)   Check with Multi device case.
    ii)  Pass stream as user created stream
    iii) Pass stream as default stream
    iv)  Pass stream as hipStreamPerThread
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraph_PerfCheck_hipGraphExecUpdate_kernel_inLoop") {
  if ((setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST(
        "Unable to turn on "
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE, hence exit!");
    return;
  }

  hipStream_t stream;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  SECTION("Multi device test with different type of stream") {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));

      SECTION("Pass stream as used created stream") {
        HIP_CHECK(hipStreamCreate(&stream));
        hipGraph_PerfCheck_hipGraphExecUpdate_kernel_inLoop(stream);
        HIP_CHECK(hipStreamDestroy(stream));
      }
      SECTION("Pass stream as default stream") {
        stream = 0;
        hipGraph_PerfCheck_hipGraphExecUpdate_kernel_inLoop(stream);
      }
      SECTION("Pass stream as hipStreamPerThread") {
        stream = hipStreamPerThread;
        hipGraph_PerfCheck_hipGraphExecUpdate_kernel_inLoop(stream);
      }
    }
  }
}
#endif
#endif  //  #if __linux__

/**
 * End doxygen group GraphTest.
 * @}
 */
