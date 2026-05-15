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
#pragma once

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <resource_guards.hh>

template <typename T> __global__ void updateResult(T* C_d, T* Res_d, T val,
                                                  int NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (int i = NELEM - stride + offset; i >= 0; i -= stride) {
    Res_d[i] = C_d[i] + val;
  }
}

template <typename T> __global__ void vectorSum(const T* A_d, const T* B_d,
                                 const T* C_d, T* Res_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    Res_d[i] = A_d[i] + B_d[i] + C_d[i];
  }
}

template <typename T>
void graphNodesCommon(hipGraph_t& graph, T* hostMem1, T* devMem1, T* hostMem2, T* devMem2,
                        T* hostMem3, T* devMem3, size_t N, std::vector<hipGraphNode_t>& from,
                        std::vector<hipGraphNode_t>& to, std::vector<hipGraphNode_t>& nodelist) {
  size_t Nbytes = N * sizeof(T);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devMem1);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(T);
  memsetParams.width = N;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));

  from.push_back(memset_A);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devMem2);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(T);
  memsetParams.width = N;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, graph, nullptr, 0, &memsetParams));

  from.push_back(memset_B);

  void* kernelArgs1[] = {&devMem3, &memsetVal, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::memsetReverse<uint32_t>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, graph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, devMem1, hostMem1, Nbytes,
                                    hipMemcpyHostToDevice));

  from.push_back(memcpyH2D_A);
  to.push_back(memcpyH2D_A);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, devMem2, hostMem2, Nbytes,
                                    hipMemcpyHostToDevice));

  from.push_back(memcpyH2D_B);
  to.push_back(memcpyH2D_B);
  from.push_back(memsetKer_C);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, hostMem3, devMem3, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&devMem1, &devMem2, &devMem3, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<T>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kernelNodeParams));

  from.push_back(kernel_vecAdd);
  to.push_back(kernel_vecAdd);
  to.push_back(kernel_vecAdd);
  to.push_back(kernel_vecAdd);
  to.push_back(memcpyD2H_C);


  nodelist.push_back(memset_A);
  nodelist.push_back(memset_B);
  nodelist.push_back(memsetKer_C);
  nodelist.push_back(memcpyH2D_A);
  nodelist.push_back(memcpyH2D_B);
  nodelist.push_back(kernel_vecAdd);
  nodelist.push_back(memcpyD2H_C);
}

template <typename T>
void captureNodesCommon(hipGraph_t& graph, T* hostMem1, T* devMem1, T* hostMem2, T* devMem2,
                          T* hostMem3, T* devMem3, size_t N, std::vector<hipStream_t>& streams,
                          std::vector<hipEvent_t>& events) {
  size_t Nbytes = N * sizeof(T);
  constexpr unsigned threadsPerBlock = 256;
  constexpr auto blocksPerCU = 6;  // to hide latency
  size_t NElem{N};
  int memsetVal{0};

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[2], events[0], 0));
  // Add operations to stream3
  hipLaunchKernelGGL(HipTest::memsetReverse<T>, dim3(blocks), dim3(threadsPerBlock), 0, streams[2],
                     devMem3, memsetVal, NElem);
  HIP_CHECK(hipEventRecord(events[1], streams[2]));
  // Add operations to stream2
  HIP_CHECK(hipMemsetAsync(devMem2, 0, Nbytes, streams[1]));
  HIP_CHECK(hipMemcpyAsync(devMem2, hostMem2, Nbytes, hipMemcpyHostToDevice, streams[1]));
  HIP_CHECK(hipEventRecord(events[2], streams[1]));
  // Add operations to stream1
  HIP_CHECK(hipMemsetAsync(devMem1, 0, Nbytes, streams[0]));
  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem1, Nbytes, hipMemcpyHostToDevice, streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[2], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[1], 0));
  hipLaunchKernelGGL(HipTest::vectorADD<T>, dim3(blocks), dim3(threadsPerBlock), 0, streams[0],
                     devMem1, devMem2, devMem3, NElem);
  HIP_CHECK(hipMemcpyAsync(hostMem3, devMem3, Nbytes, hipMemcpyDeviceToHost, streams[0]));
  HIP_CHECK(hipStreamEndCapture(streams[0], &graph));
}

enum class GraphGetNodesTest { equalNumNodes, lesserNumNodes, greaterNumNodes };

template <typename F>
static void validateGraphNodesCommon(
    F f, std::vector<hipGraphNode_t>& nodelist, size_t testNumNodes, GraphGetNodesTest test_type) {
  size_t numNodes = testNumNodes;
  hipGraphNode_t* nodes = new hipGraphNode_t[numNodes]{};
  int found_count{0};
  HIP_CHECK(f(nodes, &numNodes));
  // Count how many nodes from the nodelist are present
  for (auto node : nodelist) {
    for (size_t i = 0; i < numNodes; i++) {
      if (node == nodes[i]) {
        found_count++;
        break;
      }
    }
  }

  // Verify that the found number of nodes is expected
  switch (test_type) {
    case GraphGetNodesTest::equalNumNodes:
      REQUIRE(found_count == nodelist.size());
      break;
    case GraphGetNodesTest::lesserNumNodes:
      // Verify numNodes is unchanged
      REQUIRE(numNodes == testNumNodes);
      REQUIRE(found_count == testNumNodes);
      break;
    case GraphGetNodesTest::greaterNumNodes:
      // Verify numNodes is reset to actual number of nodes
      REQUIRE(numNodes == nodelist.size());
      REQUIRE(found_count == nodelist.size());
      // Verify additional entries in nodes are set to nullptr
      for (auto i = numNodes; i < testNumNodes; i++) {
        REQUIRE(nodes[i] == nullptr);
      }
  }
  delete[] nodes;
}
