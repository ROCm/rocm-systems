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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_kernels.hh>

#define GRAPH_LAUNCH_ITERATIONS 500
static constexpr int N = 1024;
static constexpr int Nbytes = N * sizeof(int);
static size_t NElem{N};
static constexpr int blocksPerCU = 6;  // to hide latency
static constexpr int threadsPerBlock = 256;
// Num of parallel Branches
const unsigned int kNumNode = 5;

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);`
 * - Launches an executable graph in the specified stream.
 */
unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
template <typename T> __global__ void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    C_d[i] = A_d[i] + B_d[i];
  }
}
/**
 * Test Description
 * ------------------------
 * - Create the graph with multiple parallel branches.
 * - Calculate the time taken to graph execution.
 * Test source
 * ------------------------
 * - catch\perftests\graph\parallelGraph.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraph_Performance_Improvement_ParallelGraph") {
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
    kernelNodeParams.func = reinterpret_cast<void*>(vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0, &kernelNodeParams));
    HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  for (int i = 0; i < kNumNode; i++) {
    HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i], &memCpy3, 1));
  }

  hipGraphNode_t* nodes{nullptr};
  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  INFO("Num of nodes in the graph: " << numNodes);
  INFO("Number Graph Launches: " << GRAPH_LAUNCH_ITERATIONS);
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  // start time
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  // Stop time
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = stop - start;
  INFO("Time taken for Graph: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
       << " milliSeconds");
  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Enque some operations on to a stream.
 * - Operatoions should be same as above test graph nodes.
 * - Calculate the time taken to complete stream operations.
 * Test source
 * ------------------------
 * - catch\perftests\graph\parallelGraph.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraph_Performance_With_Stream_Operations") {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyDefault, stream));
    HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyDefault, stream));
    for (int i = 0; i < kNumNode; i++) {
      hipLaunchKernelGGL(vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d, B_d, C_d,
                         NElem);
    }
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDefault, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = stop - start;
  INFO("Time taken for Stream: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
       << " milliSeconds");
  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Create a graph via Stream capture.
 * - Calculate the time taken to complete the graph execution.
 * Test source
 * ------------------------
 * - catch\perftests\graph\parallelGraph.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */

TEST_CASE("Unit_hipGraph_Performance_With_Stream_Capture") {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraph_t graph;
  hipStream_t stream, streamForGraph;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyDefault, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyDefault, stream));
  for (int i = 0; i < kNumNode; i++) {
    hipLaunchKernelGGL(vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d, B_d, C_d,
                       NElem);
  }
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDefault, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  }
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = stop - start;
  INFO("Time taken for Graph via Stream Capture: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
       << " milliSeconds");
  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
