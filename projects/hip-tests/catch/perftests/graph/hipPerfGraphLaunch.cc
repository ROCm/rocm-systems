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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

static constexpr int launches = 5;

/**
 * The tests in this file are added to see the performance improvement with the
 * Alloc node detection optimization task : SWDEV-490864
 */

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);`
 * - Launches an executable graph in the specified stream.
 */

/**
 * In fillKernel, all elements of the array filled with given value
 */
static __global__ void fillKernel(int* arr, int size, int value) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i += stride) {
    arr[i] = value;
  }
}

/**
 * In addOneKernel, all elements of the array are incremented by 1
 */
static __global__ void addOneKernel(int* arr, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i += stride) {
    arr[i] += 1;
  }
}

/**
 * In addKernel, Array1 and Array2 will be added by element wise
 * and stored in Array 1
 */
static __global__ void addKernel(int* arr1, int* arr2, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i += stride) {
    arr1[i] = arr1[i] + arr2[i];
  }
}

/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Create 1024 Mem alloc Nodes, make them serial dependent.
 * -    (Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 2) Create 1024 Mem free Nodes.
 * -    (Node 0 depends on last created mem alloc node,
 * -     Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 3) Launch the graph repeatedly
 * - 4) Capture the Graph exection time and Synchronization time.
 *
 * Test source
 * ------------------------
 * - catch/perftests/graph/hipPerfGraphLaunch.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Perf_GraphWithMoreAllocFreeNodes_SingleBranchNoOperations") {
  constexpr int numberOfNodes = 1024;

  int* devMem[numberOfNodes];
  for (int i = 0; i < numberOfNodes; i++) {
    devMem[i] = nullptr;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode[numberOfNodes], memFreeNode[numberOfNodes];

  // Prapare Mem Alloc Nodes
  for (int i = 0; i < numberOfNodes; i++) {
    hipMemAllocNodeParams memAllocNodeParams{};
    memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
    memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
    memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
    memAllocNodeParams.poolProps.location.id = 0;
    memAllocNodeParams.bytesize = sizeof(int);

    if (i == 0) {
      HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[i], graph, nullptr, 0, &memAllocNodeParams));
    } else {
      ::std::vector<hipGraphNode_t> memAllocNodeDependencies;
      memAllocNodeDependencies.push_back(memAllocNode[i - 1]);

      HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[i], graph, memAllocNodeDependencies.data(),
                                        memAllocNodeDependencies.size(), &memAllocNodeParams));
    }
    devMem[i] = reinterpret_cast<int*>(memAllocNodeParams.dptr);
    REQUIRE(devMem[i] != nullptr);
  }

  // Prapare Mem Free Nodes
  for (int i = 0; i < numberOfNodes; i++) {
    if (i == 0) {
      ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
      memFreeNodeDependencies.push_back(memAllocNode[numberOfNodes - 1]);

      HIP_CHECK(hipGraphAddMemFreeNode(&memFreeNode[i], graph, memFreeNodeDependencies.data(),
                                       memFreeNodeDependencies.size(),
                                       reinterpret_cast<void*>(devMem[i])));
    } else {
      ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
      memFreeNodeDependencies.push_back(memFreeNode[i - 1]);

      HIP_CHECK(hipGraphAddMemFreeNode(&memFreeNode[i], graph, memFreeNodeDependencies.data(),
                                       memFreeNodeDependencies.size(),
                                       reinterpret_cast<void*>(devMem[i])));
    }
  }

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));

  // Warm up call
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "Graph launches = " << launches << std::endl;

  auto launch_start = std::chrono::high_resolution_clock::now();

  for (int itr = 1; itr <= launches; itr++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }

  auto launch_stop = std::chrono::high_resolution_clock::now();
  auto launch_result = std::chrono::duration<double, std::milli>(launch_stop - launch_start);

  auto sync_start = std::chrono::high_resolution_clock::now();

  HIP_CHECK(hipStreamSynchronize(stream));

  auto sync_stop = std::chrono::high_resolution_clock::now();
  auto sync_result = std::chrono::duration<double, std::milli>(sync_stop - sync_start);

  std::cout << "Time taken to Execute : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(launch_result).count()
            << " millisecs " << std::endl;

  std::cout << "Time taken to Synchronize : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(sync_result).count()
            << " millisecs " << std::endl;

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Create 100 Mem alloc Nodes, make them serial dependent.
 * -    (Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 2) Create 100 Memset Nodes.
 * -    (Node 0 depends on last created mem alloc node,
 * -     Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 3) Create 100 Kernel Nodes.
 * -    (Node 0 depends on last created mem set node,
 * -     Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 4) Create 100 Memcpy Nodes.
 * -    (Node 0 depends on last created kernel node,
 * -     Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 5) Create 100 Mem free Nodes.
 * -    (Node 0 depends on last created mem copy node,
 * -     Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 6) Launch the graph repeatedly
 * - 7) Capture the Graph exection time and Synchronization time.
 *
 * Test source
 * ------------------------
 * - catch/perftests/graph/hipPerfGraphLaunch.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Perf_GraphWithMoreAllocFreeNodes_SerialNodesSingleBranchWithOps") {
  constexpr int SIZE = 100;

  char* dev[SIZE];
  for (int i = 0; i < SIZE; i++) {
    dev[i] = nullptr;
  }

  char hostDst[SIZE];
  for (int i = 0; i < SIZE; i++) {
    hostDst[i] = 0;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode[SIZE], memsetNode[SIZE], kernelNode[SIZE], memcpyNode[SIZE],
      memFreeNode[SIZE];

  // Prapare Mem alloc Nodes
  for (int i = 0; i < SIZE; i++) {
    hipMemAllocNodeParams memAllocNodeParams{};
    memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
    memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
    memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
    memAllocNodeParams.poolProps.location.id = 0;
    memAllocNodeParams.bytesize = sizeof(char);

    if (i == 0) {
      HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[i], graph, nullptr, 0, &memAllocNodeParams));
    } else {
      ::std::vector<hipGraphNode_t> memAllocNodeDependencies;
      memAllocNodeDependencies.push_back(memAllocNode[i - 1]);

      HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[i], graph, memAllocNodeDependencies.data(),
                                        memAllocNodeDependencies.size(), &memAllocNodeParams));
    }
    dev[i] = reinterpret_cast<char*>(memAllocNodeParams.dptr);
    REQUIRE(dev[i] != nullptr);
  }

  // Prapare Memset Nodes
  for (int i = 0; i < SIZE; i++) {
    hipMemsetParams pMemsetParams{};
    pMemsetParams.dst = reinterpret_cast<void*>(dev[i]);
    pMemsetParams.elementSize = 1;
    pMemsetParams.height = 1;
    pMemsetParams.pitch = 1;
    pMemsetParams.value = i;
    pMemsetParams.width = 1;

    ::std::vector<hipGraphNode_t> memsetNodeDependencies;
    if (i == 0) {
      memsetNodeDependencies.push_back(memAllocNode[SIZE - 1]);
    } else {
      memsetNodeDependencies.push_back(memsetNode[i - 1]);
    }
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode[i], graph, memsetNodeDependencies.data(),
                                    memsetNodeDependencies.size(), &pMemsetParams));
  }

  // Prapare Kernel Nodes
  for (int i = 0; i < SIZE; i++) {
    hipKernelNodeParams kernelNodeParams{};
    kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(1, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    int size = 1;
    void* kernelArgs[2] = {reinterpret_cast<void*>(&dev[i]), reinterpret_cast<void*>(&size)};
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = nullptr;

    ::std::vector<hipGraphNode_t> kernelNodeDependencies;
    if (i == 0) {
      kernelNodeDependencies.push_back(memsetNode[SIZE - 1]);
    } else {
      kernelNodeDependencies.push_back(kernelNode[i - 1]);
    }

    HIP_CHECK(hipGraphAddKernelNode(&kernelNode[i], graph, kernelNodeDependencies.data(),
                                    kernelNodeDependencies.size(), &kernelNodeParams));
  }

  // Prapare Memcpy Nodes
  for (int i = 0; i < SIZE; i++) {
    hipMemcpy3DParms pMemcpyParams{};
    pMemcpyParams.srcPos = make_hipPos(0, 0, 0);
    pMemcpyParams.dstPos = make_hipPos(0, 0, 0);
    pMemcpyParams.srcPtr = make_hipPitchedPtr(dev[i], 1, 1, 1);
    pMemcpyParams.dstPtr = make_hipPitchedPtr(&hostDst[i], 1, 1, 1);
    pMemcpyParams.extent = make_hipExtent(1, 1, 1);
    pMemcpyParams.kind = hipMemcpyDeviceToHost;

    ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
    if (i == 0) {
      memcpyNodeDependencies.push_back(kernelNode[SIZE - 1]);
    } else {
      memcpyNodeDependencies.push_back(memcpyNode[i - 1]);
    }
    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode[i], graph, memcpyNodeDependencies.data(),
                                    memcpyNodeDependencies.size(), &pMemcpyParams));
  }

  // Prapare Mem free Nodes
  for (int i = 0; i < SIZE; i++) {
    if (i == 0) {
      ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
      memFreeNodeDependencies.push_back(memcpyNode[SIZE - 1]);

      HIP_CHECK(hipGraphAddMemFreeNode(&memFreeNode[i], graph, memFreeNodeDependencies.data(),
                                       memFreeNodeDependencies.size(),
                                       reinterpret_cast<void*>(dev[i])));
    } else {
      ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
      memFreeNodeDependencies.push_back(memFreeNode[i - 1]);

      HIP_CHECK(hipGraphAddMemFreeNode(&memFreeNode[i], graph, memFreeNodeDependencies.data(),
                                       memFreeNodeDependencies.size(),
                                       reinterpret_cast<void*>(dev[i])));
    }
  }

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));

  // Warm up call
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "Graph launches = " << launches << std::endl;

  auto launch_start = std::chrono::high_resolution_clock::now();

  for (int itr = 1; itr <= launches; itr++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }
  auto launch_stop = std::chrono::high_resolution_clock::now();
  auto launch_result = std::chrono::duration<double, std::milli>(launch_stop - launch_start);

  auto sync_start = std::chrono::high_resolution_clock::now();

  HIP_CHECK(hipStreamSynchronize(stream));

  auto sync_stop = std::chrono::high_resolution_clock::now();
  auto sync_result = std::chrono::duration<double, std::milli>(sync_stop - sync_start);

  std::cout << "Time taken to Execute : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(launch_result).count()
            << " millisecs " << std::endl;

  std::cout << "Time taken to Synchronize : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(sync_result).count()
            << " millisecs " << std::endl;

  for (int i = 0; i < SIZE; i++) {
    REQUIRE(hostDst[i] == (i + 1));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Create memcpy Node ( H2D ) as a root node
 * - 2) Create 10 Mem Alloc nodes, 10 Fill Kernel Nodes, 10 Add Kernel Nodes
 * -    and 10 memcpy Nodes (D2H).
 * -    a) All Mem Alloc nodes depends memcpy H2D Node.
 * -       ( Mem Alloc Node 0 to 9, depends on memcpy H2D Node)
 * -    b) Fill Kernel Node 0 depends on Mem Alloc Node 0,
 * -       Fill Kernel Node 1 depends on Mem Alloc Node 1, and so on.
 * -    c) Add Kernel Node 0 depends on Fill Kernel Node 0,
 * -       Add Kernel Node 1 depends on Fill Kernel Node 1, and so on.
 * -    d) MemcpyD2H Node 0 depends on Add Kernel Node 0,
 * -       MemcpyD2H Node 1 depends on Add Kernel Node 1, and so on.
 * - 3) Create MemcpyH2H, which depend on all the 10 MemcpyD2H Nodes
 * - 4) Launch the graph repeatedly
 * - 5) Capture the Graph exection time and Synchronization time.
 *
 * Test source
 * ------------------------
 * - catch/perftests/graph/hipPerfGraphLaunch.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Perf_GraphWithMoreAllocFreeNodes_MultipleBranches") {
  constexpr int SIZE = 1024;
  constexpr size_t NBYTES = SIZE * sizeof(int);
  constexpr int BRANCHES = 10;

  int value = 100;
  int* hostMemSrc = new int[SIZE];
  REQUIRE(hostMemSrc != nullptr);

  int* devMemSrc1 = nullptr;
  HIP_CHECK(hipMalloc(&devMemSrc1, NBYTES));
  REQUIRE(devMemSrc1 != nullptr);

  int* devMemSrc2[BRANCHES];
  for (int i = 0; i < BRANCHES; i++) {
    devMemSrc2[i] = nullptr;
  }

  int hostMemDst[BRANCHES][SIZE];
  int finalHostDst[BRANCHES * SIZE];

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyNodeH2D, memAllocNode[BRANCHES], fillKernelNode[BRANCHES],
      addKernelNode[BRANCHES], memcpyNodeD2H[BRANCHES], memFreeNode[BRANCHES], memcpyNodeH2H;

  // Add H2D Node
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0, devMemSrc1, hostMemSrc,
                                    NBYTES, hipMemcpyHostToDevice));

  for (int branch = 0; branch < BRANCHES; branch++) {
    // Add Mem alloc Nodes
    ::std::vector<hipGraphNode_t> memAllocNodeDependencies;
    memAllocNodeDependencies.push_back(memcpyNodeH2D);

    hipMemAllocNodeParams memAllocNodeParams{};
    memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
    memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
    memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
    memAllocNodeParams.poolProps.location.id = 0;
    memAllocNodeParams.bytesize = NBYTES;

    HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[branch], graph, memAllocNodeDependencies.data(),
                                      memAllocNodeDependencies.size(), &memAllocNodeParams));

    devMemSrc2[branch] = reinterpret_cast<int*>(memAllocNodeParams.dptr);
    REQUIRE(devMemSrc2[branch] != nullptr);

    // Add Kernel Nodes (fillKernel)
    ::std::vector<hipGraphNode_t> kernelNodeDependencies;
    kernelNodeDependencies.push_back(memAllocNode[branch]);

    hipKernelNodeParams kernelNodeParams{};
    kernelNodeParams.func = reinterpret_cast<void*>(fillKernel);
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(1, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    int size = SIZE;
    void* kernelArgs[3] = {reinterpret_cast<void*>(&devMemSrc2[branch]),
                           reinterpret_cast<void*>(&size), reinterpret_cast<void*>(&value)};
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&fillKernelNode[branch], graph, kernelNodeDependencies.data(),
                                    kernelNodeDependencies.size(), &kernelNodeParams));

    // Add Kernel Nodes (addKernel)
    ::std::vector<hipGraphNode_t> kernelNodeDependencies2;
    kernelNodeDependencies2.push_back(fillKernelNode[branch]);

    hipKernelNodeParams kernelNodeParams2{};
    kernelNodeParams2.func = reinterpret_cast<void*>(addKernel);
    kernelNodeParams2.gridDim = dim3(1, 1, 1);
    kernelNodeParams2.blockDim = dim3(1, 1, 1);
    kernelNodeParams2.sharedMemBytes = 0;
    int size2 = SIZE;
    void* kernelArgs2[3] = {reinterpret_cast<void*>(&devMemSrc2[branch]),
                            reinterpret_cast<void*>(&devMemSrc1), reinterpret_cast<void*>(&size2)};
    kernelNodeParams2.kernelParams = kernelArgs2;
    kernelNodeParams2.extra = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&addKernelNode[branch], graph, kernelNodeDependencies2.data(),
                                    kernelNodeDependencies2.size(), &kernelNodeParams2));

    // Add D2H Nodes
    ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
    memcpyNodeD2HDependencies.push_back(addKernelNode[branch]);

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H[branch], graph,
                                      memcpyNodeD2HDependencies.data(),
                                      memcpyNodeD2HDependencies.size(), hostMemDst[branch],
                                      devMemSrc2[branch], NBYTES, hipMemcpyDeviceToHost));

    ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
    memFreeNodeDependencies.push_back(memcpyNodeD2H[branch]);

    HIP_CHECK(hipGraphAddMemFreeNode(&memFreeNode[branch], graph, memFreeNodeDependencies.data(),
                                     memFreeNodeDependencies.size(),
                                     reinterpret_cast<void*>(devMemSrc2[branch])));
  }

  // Add H2H Node
  ::std::vector<hipGraphNode_t> memcpyNodeH2HDependencies;
  for (int i = 0; i < BRANCHES; i++) {
    memcpyNodeH2HDependencies.push_back(memFreeNode[i]);
  }

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2H, graph, memcpyNodeH2HDependencies.data(),
                                    memcpyNodeH2HDependencies.size(), finalHostDst, hostMemDst,
                                    BRANCHES * SIZE * sizeof(int), hipMemcpyHostToHost));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));

  // Warm up call
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "Graph launches : " << launches << std::endl;

  auto launch_start = std::chrono::high_resolution_clock::now();

  for (int launch = 1; launch <= launches; launch++) {
    std::fill(hostMemSrc, hostMemSrc + SIZE, launch);

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }

  auto launch_stop = std::chrono::high_resolution_clock::now();
  auto launch_result = std::chrono::duration<double, std::milli>(launch_stop - launch_start);

  auto sync_start = std::chrono::high_resolution_clock::now();

  HIP_CHECK(hipStreamSynchronize(stream));

  auto sync_stop = std::chrono::high_resolution_clock::now();
  auto sync_result = std::chrono::duration<double, std::milli>(sync_stop - sync_start);

  std::cout << "Time taken to Execute : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(launch_result).count()
            << " millisecs " << std::endl;

  std::cout << "Time taken to Synchronize : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(sync_result).count()
            << " millisecs " << std::endl;

  for (int branch = 0; branch < BRANCHES; branch++) {
    for (int idx = 0; idx < SIZE; idx++) {
      REQUIRE(hostMemDst[branch][idx] == (launches + value));
    }
  }
  for (int idx = 0; idx < BRANCHES * SIZE; idx++) {
    REQUIRE(finalHostDst[idx] == (launches + value));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMemSrc1));
}

/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Create mem alloc Node
 * - 2) Create memset Node, which depends on mem alloc Node
 * - 3) Create kernel Node, which depends on memset Node
 * - 4) Create memcpy Node, which depends on kernel Node
 * - 5) Create Mem free Node, which depends on memcpy Node
 * - 6) Repeat the above 5 steps 10 times, and graph will be created with
 * -    the 10 independent branches.
 * - 7) Launch the graph repeatedly
 * - 8) Capture the Graph exection time and Synchronization time.
 *
 * Test source
 * ------------------------
 * - catch/perftests/graph/hipPerfGraphLaunch.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Perf_GraphWithMoreAllocFreeNodes_MultipleIndependentBranches") {
  constexpr int BRANCHES = 10;

  char* dev[BRANCHES];
  for (int i = 0; i < BRANCHES; i++) {
    dev[i] = nullptr;
  }

  char hostDst[BRANCHES];
  for (int i = 0; i < BRANCHES; i++) {
    hostDst[i] = 0;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode[BRANCHES], memsetNode[BRANCHES], kernelNode[BRANCHES],
      memcpyNode[BRANCHES], memFreeNode[BRANCHES];

  // Prapare Mem alloc Nodes
  for (int i = 0; i < BRANCHES; i++) {
    hipMemAllocNodeParams memAllocNodeParams{};
    memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
    memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
    memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
    memAllocNodeParams.poolProps.location.id = 0;
    memAllocNodeParams.bytesize = sizeof(char);

    HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[i], graph, nullptr, 0, &memAllocNodeParams));

    dev[i] = reinterpret_cast<char*>(memAllocNodeParams.dptr);
    REQUIRE(dev[i] != nullptr);

    hipMemsetParams pMemsetParams{};
    pMemsetParams.dst = reinterpret_cast<void*>(dev[i]);
    pMemsetParams.elementSize = 1;
    pMemsetParams.height = 1;
    pMemsetParams.pitch = 1;
    pMemsetParams.value = i;
    pMemsetParams.width = 1;

    ::std::vector<hipGraphNode_t> memsetNodeDependencies;
    memsetNodeDependencies.push_back(memAllocNode[i]);

    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode[i], graph, memsetNodeDependencies.data(),
                                    memsetNodeDependencies.size(), &pMemsetParams));

    hipKernelNodeParams kernelNodeParams{};
    kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(1, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    int size = 1;
    void* kernelArgs[2] = {reinterpret_cast<void*>(&dev[i]), reinterpret_cast<void*>(&size)};
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = nullptr;

    ::std::vector<hipGraphNode_t> kernelNodeDependencies;
    kernelNodeDependencies.push_back(memsetNode[i]);

    HIP_CHECK(hipGraphAddKernelNode(&kernelNode[i], graph, kernelNodeDependencies.data(),
                                    kernelNodeDependencies.size(), &kernelNodeParams));

    hipMemcpy3DParms pMemcpyParams{};
    pMemcpyParams.srcPos = make_hipPos(0, 0, 0);
    pMemcpyParams.dstPos = make_hipPos(0, 0, 0);
    pMemcpyParams.srcPtr = make_hipPitchedPtr(dev[i], 1, 1, 1);
    pMemcpyParams.dstPtr = make_hipPitchedPtr(&hostDst[i], 1, 1, 1);
    pMemcpyParams.extent = make_hipExtent(1, 1, 1);
    pMemcpyParams.kind = hipMemcpyDeviceToHost;

    ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
    memcpyNodeDependencies.push_back(kernelNode[i]);

    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode[i], graph, memcpyNodeDependencies.data(),
                                    memcpyNodeDependencies.size(), &pMemcpyParams));

    ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
    memFreeNodeDependencies.push_back(memcpyNode[i]);

    HIP_CHECK(hipGraphAddMemFreeNode(&memFreeNode[i], graph, memFreeNodeDependencies.data(),
                                     memFreeNodeDependencies.size(),
                                     reinterpret_cast<void*>(dev[i])));
  }

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));

  // Warm up call
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "Graph launches = " << launches << std::endl;

  auto launch_start = std::chrono::high_resolution_clock::now();

  for (int itr = 1; itr <= launches; itr++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }
  auto launch_stop = std::chrono::high_resolution_clock::now();
  auto launch_result = std::chrono::duration<double, std::milli>(launch_stop - launch_start);

  auto sync_start = std::chrono::high_resolution_clock::now();

  HIP_CHECK(hipStreamSynchronize(stream));

  auto sync_stop = std::chrono::high_resolution_clock::now();
  auto sync_result = std::chrono::duration<double, std::milli>(sync_stop - sync_start);

  std::cout << "Time taken to Execute : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(launch_result).count()
            << " millisecs " << std::endl;

  std::cout << "Time taken to Synchronize : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(sync_result).count()
            << " millisecs " << std::endl;

  for (int i = 0; i < BRANCHES; i++) {
    REQUIRE(hostDst[i] == (i + 1));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Create 1024 Mem alloc Nodes. And make them serial dependent.
 * -    (Node 1 depends on Node 0, Node 2 depends on Node 1, and so on)
 * - 2) Instantiate the graph with hipGraphInstantiateFlagAutoFreeOnLaunch flag
 * - 3) Launch the graph repeatedly
 * - 4) Capture the Graph exection time and Synchronization time.
 *
 * Test source
 * ------------------------
 * - catch/perftests/graph/hipPerfGraphLaunch.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Perf_GraphWithMoreAllocFreeNodes_OneBranchNoOps_AutoFreeOnLaunch") {
  constexpr int SIZE = 1024;

  int* devMem[SIZE];
  for (int i = 0; i < SIZE; i++) {
    devMem[i] = nullptr;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode[SIZE];

  for (int i = 0; i < SIZE; i++) {
    hipMemAllocNodeParams memAllocNodeParams{};
    memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
    memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
    memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
    memAllocNodeParams.poolProps.location.id = 0;
    memAllocNodeParams.bytesize = sizeof(int);

    if (i == 0) {
      HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[i], graph, nullptr, 0, &memAllocNodeParams));
    } else {
      ::std::vector<hipGraphNode_t> memAllocNodeDependencies;
      memAllocNodeDependencies.push_back(memAllocNode[i - 1]);

      HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode[i], graph, memAllocNodeDependencies.data(),
                                        memAllocNodeDependencies.size(), &memAllocNodeParams));
    }
    devMem[i] = reinterpret_cast<int*>(memAllocNodeParams.dptr);
    REQUIRE(devMem[i] != nullptr);
  }

  hipGraphExec_t graphExec;
  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

  // Warm up call
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "Graph launches = " << launches << std::endl;

  auto launch_start = std::chrono::high_resolution_clock::now();

  for (int itr = 1; itr <= launches; itr++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }

  auto launch_stop = std::chrono::high_resolution_clock::now();
  auto launch_result = std::chrono::duration<double, std::milli>(launch_stop - launch_start);

  auto sync_start = std::chrono::high_resolution_clock::now();

  HIP_CHECK(hipStreamSynchronize(stream));

  auto sync_stop = std::chrono::high_resolution_clock::now();
  auto sync_result = std::chrono::duration<double, std::milli>(sync_stop - sync_start);

  std::cout << "Time taken to Execute : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(launch_result).count()
            << " millisecs " << std::endl;

  std::cout << "Time taken to Synchronize : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(sync_result).count()
            << " millisecs " << std::endl;

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
