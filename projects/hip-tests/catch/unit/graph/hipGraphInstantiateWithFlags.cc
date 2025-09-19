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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph, unsigned long long
flags); Testcase Scenarios of hipGraphInstantiateWithFlags API: Negative: 1) Pass nullptr to
pGraphExec 2) Pass nullptr to graph 4) Pass invalid flag Functional: 1) Create dependencies graph
and instantiate the graph 2) Create graph in one GPU device and instantiate, launch in peer GPU
device 3) Create stream capture graph and instantite the graph 4) Create stream capture graph in one
GPU device  and instantite the graph launch in peer GPU device Mapping is missing for NVIDIA
platform hence skipping the testcases
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#endif

constexpr size_t N = 1000000;
static constexpr int SIZE = 1024 * 1024;
static constexpr size_t NBYTES = SIZE * sizeof(int);

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
 * In doubleKernel, all elements of the array doubled with its value
 */
static __global__ void doubleKernel(int* arr, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i += stride) {
    arr[i] += arr[i];
  }
}

/* This test covers the negative scenarios of
   hipGraphInstantiateWithFlags API */
TEST_CASE("Unit_hipGraphInstantiateWithFlags_Negative") {
  SECTION("Passing nullptr pGraphExec") {
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    REQUIRE(hipGraphInstantiateWithFlags(nullptr, graph, 0) == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    hipGraphExec_t graphExec;
    REQUIRE(hipGraphInstantiateWithFlags(&graphExec, nullptr, 0) == hipErrorInvalidValue);
  }

  SECTION("Passing Invalid flag") {
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphExec_t graphExec;
    REQUIRE(hipGraphInstantiateWithFlags(&graphExec, graph, 10) != hipSuccess);
  }
}
/*
This function verifies the following scenarios
1. Creates dependency graph, Instantiates the graph with flags and verifies it
2. Creates graph on one GPU-1 device and instantiates the graph on peer GPU device
*/
void GraphInstantiateWithFlags_DependencyGraph(bool ctxt_change = false) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, graph, nullptr, 0, &memsetParams));

  void* kernelArgs1[] = {&C_d, &memsetVal, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::memsetReverse<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, graph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetKer_C, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));

  if (ctxt_change) {
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
  }
  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/*
This function verifies the following scenarios
1. Creates stream capture graph, Instantiates the graph with flags and verifies it
2. Creates graph on one GPU-1 device and instantiates the graph on peer GPU device
*/
void GraphInstantiateWithFlags_StreamCapture(bool deviceContextChg = false) {
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t Nbytes = N * sizeof(float);
  hipStream_t stream;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
  }
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);
  HIP_CHECK(hipGraphCreate(&graph, 0));


  HIP_CHECK(hipStreamCreate(&stream));
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  if (deviceContextChg) {
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
  }

  // Validate end capture is successful
  REQUIRE(graph != nullptr);
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
  REQUIRE(graphExec != nullptr);

  HIP_CHECK(hipGraphLaunch(graphExec, stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("A and C not matching at " << i);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}
/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating dependency graph and instantiate, launching and verifying
the result
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_DependencyGraph") {
  GraphInstantiateWithFlags_DependencyGraph();
}

/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating dependency graph on GPU-0 and instantiate, launching and verifying
the result on GPU-1
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_DependencyGraphDeviceCtxtChg") {
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      GraphInstantiateWithFlags_DependencyGraph(true);
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating capture graph and instantiate, launching and verifying
the result
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_StreamCapture") {
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      GraphInstantiateWithFlags_StreamCapture();
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating capture graph on GPU-0 and instantiate, launching and verifying
the result on GPU-1
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_StreamCaptureDeviceContextChg") {
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      GraphInstantiateWithFlags_StreamCapture(true);
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

/* Create graph and add memAlloc node, but no corresponding memFree node to it.
  Instantiate graph with flag - hipGraphInstantiateFlagAutoFreeOnLaunch
  Launch and check graph execution should work properly and
  free memory allocated by memAlloc call manually using hipFree api.

Note - This test case is just to check if hipGraphInstantiateFlagAutoFreeOnLaunch
       is not resulting in compilation error or api failure. Real functional test
       will be added once the feature is fully implemented.
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_FlagAutoFreeOnLaunch_check") {
  constexpr size_t size = 512 * 1024 * 1024;
  constexpr size_t Nbytes = size * sizeof(int);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t allocNodeA;
  hipMemAllocNodeParams allocParam;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam));
  REQUIRE(allocParam.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(allocParam.dptr);

  // Instantiate with Flag and launch the graph
  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  size_t bmem = 0, bmemres = 0;
  HIP_CHECK(hipDeviceGraphMemTrim(0));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &bmem));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrReservedMemCurrent, &bmemres));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  size_t amem = 0, amemres = 0;
  HIP_CHECK(hipDeviceGraphMemTrim(0));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &amem));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrReservedMemCurrent, &amemres));

  REQUIRE(bmem == amem);
  REQUIRE(bmemres == amemres);

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipDeviceGraphMemTrim(0));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &amem));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrReservedMemCurrent, &amemres));

  REQUIRE(bmem == amem);
  REQUIRE(bmemres == amemres);

  HIP_CHECK(hipFree(A_d));  //  free allocMemory manually
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - This test case tests hipGraphInstantiateWithFlags with the flag
 * - hipGraphInstantiateFlagAutoFreeOnLaunch with below scenario :
 * - 1) Create graph with one MemAllocNode for 1GB of memory
 * - 2) Launch it in a loop, it should not give
 *      any memory related issues/errors.
 * Test source
 * ------------------------
 * - unit/graph/hipGraphInstantiateWithFlags.cc
 */
TEST_CASE("Unit_hipGraphInstantiateWithFlags_AutoFreeOnLaunchInLoop") {
  constexpr size_t NBytes = 1024 * 1024 * 1024;

  void* devMem = nullptr;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode;

  hipMemAllocNodeParams memAllocNodeParams{};
  memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
  memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
  memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
  memAllocNodeParams.poolProps.location.id = 0;
  memAllocNodeParams.bytesize = NBytes;

  HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode, graph, nullptr, 0, &memAllocNodeParams));
  devMem = memAllocNodeParams.dptr;

  hipGraphExec_t graphExec;
  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

  // Launch the graph in a loop
  for (int i = 0; i < 100; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(devMem != nullptr);

#if HT_AMD
    size_t sizeToCheck = -1;
    HIP_CHECK(hipMemPtrGetInfo(devMem, &sizeToCheck));
    REQUIRE(sizeToCheck == NBytes);
#endif
  }

  HIP_CHECK(hipFree(devMem));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - This test case tests hipGraphInstantiateWithFlags with the flag
 * - hipGraphInstantiateFlagAutoFreeOnLaunch for following scenario :
 * - 1) Create graph with following nodes,
 * -    a. Node to allocate memory - memAllocNode
 * -    b. Node to fill the allocated device memory - kernelNode
 * -    c. Node to copy from device to host - memcpyNodeD2H
 * - 2) Launch the graph in a loop. Check the host memory, it should not always
 * -    contain the expected value and it should not give memory related issues
 * Test source
 * ------------------------
 * - unit/graph/hipGraphInstantiateWithFlags.cc
 */
TEST_CASE("Unit_hipGraphInstantiateWithFlags_AutoFreeOnLaunchFillKernel") {
  int value = 100;

  int* hostMemDst = new int[SIZE];
  REQUIRE(hostMemDst != nullptr);
  std::fill(hostMemDst, hostMemDst + SIZE, 0);

  int* devMem = nullptr;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode, kernelNode, memcpyNodeD2H;

  hipMemAllocNodeParams memAllocNodeParams{};
  memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
  memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
  memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
  memAllocNodeParams.poolProps.location.id = 0;
  memAllocNodeParams.bytesize = NBYTES;

  HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode, graph, nullptr, 0, &memAllocNodeParams));
  devMem = reinterpret_cast<int*>(memAllocNodeParams.dptr);
  REQUIRE(devMem != nullptr);

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memAllocNode);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(fillKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  int size = SIZE;
  void* kernelArgs[3] = {reinterpret_cast<void*>(&devMem), reinterpret_cast<void*>(&size),
                         reinterpret_cast<void*>(&value)};
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, kernelNodeDependencies.data(),
                                  kernelNodeDependencies.size(), &kernelNodeParams));

  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph, memcpyNodeD2HDependencies.data(),
                                    memcpyNodeD2HDependencies.size(), hostMemDst, devMem, NBYTES,
                                    hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec;
  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

  for (int launch = 1; launch <= 10; launch++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    for (int idx = 0; idx < SIZE; idx++) {
      INFO("For Launch : " << launch << ", At index : " << idx << ", Got value : "
                           << hostMemDst[idx] << ", Expected value : " << value << "\n");
      REQUIRE(hostMemDst[idx] == value);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
  delete[] hostMemDst;
}

/**
 * Test Description
 * ------------------------
 * - This test case tests hipGraphInstantiateWithFlags with the flag
 * - hipGraphInstantiateFlagAutoFreeOnLaunch for following scenario :
 * - 1) Take a host memory
 * - 2) Create graph with following nodes,
 * -    a. Node to allocate memory - memAllocNode
 * -    b. Node to copy from host to device - memcpyNodeH2D
 * -    c. Node to perform double operation - kernelNode
 * -    d. Node to copy from device to host - memcpyNodeD2H
 * - 3) Launch the graph in a loop and for each iteration
 * -    fill new value in hostMemSrc.
 * - 4) Each time hostMemDst should contain the expected value
 * -    and it should not give memory related issues.
 * Test source
 * ------------------------
 * - unit/graph/hipGraphInstantiateWithFlags.cc
 */
TEST_CASE("Unit_hipGraphInstantiateWithFlags_AutoFreeOnLaunchDoubleKernel") {
  int* hostMemSrc = new int[SIZE];
  REQUIRE(hostMemSrc != nullptr);

  int* hostMemDst = new int[SIZE];
  REQUIRE(hostMemDst != nullptr);

  int* devMem = nullptr;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode, memcpyNodeH2D, kernelNode, memcpyNodeD2H;

  hipMemAllocNodeParams memAllocNodeParams{};
  memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
  memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
  memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
  memAllocNodeParams.poolProps.location.id = 0;
  memAllocNodeParams.bytesize = NBYTES;

  HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode, graph, nullptr, 0, &memAllocNodeParams));
  devMem = reinterpret_cast<int*>(memAllocNodeParams.dptr);
  REQUIRE(devMem != nullptr);

  ::std::vector<hipGraphNode_t> memcpyNodeH2DDependencies;
  memcpyNodeH2DDependencies.push_back(memAllocNode);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, memcpyNodeH2DDependencies.data(),
                                    memcpyNodeH2DDependencies.size(), devMem, hostMemSrc, NBYTES,
                                    hipMemcpyHostToDevice));

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(doubleKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  int size = SIZE;
  void* kernelArgs[2] = {reinterpret_cast<void*>(&devMem), reinterpret_cast<void*>(&size)};
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, kernelNodeDependencies.data(),
                                  kernelNodeDependencies.size(), &kernelNodeParams));

  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph, memcpyNodeD2HDependencies.data(),
                                    memcpyNodeD2HDependencies.size(), hostMemDst, devMem, NBYTES,
                                    hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec;
  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

  for (int launch = 1; launch <= 10; launch++) {
    std::fill(hostMemSrc, hostMemSrc + SIZE, launch);
    std::fill(hostMemDst, hostMemDst + SIZE, 0);

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    for (int idx = 0; idx < SIZE; idx++) {
      INFO("For Launch : " << launch << ", At index : " << idx
                           << ", Got value : " << hostMemDst[idx]
                           << ", Expected value : " << (launch + launch) << "\n");
      REQUIRE(hostMemDst[idx] == (launch + launch));
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
  delete[] hostMemSrc;
  delete[] hostMemDst;
}

/**
 * Test Description
 * ------------------------
 * - This test case tests hipGraphInstantiateWithFlags with the flag 0
 * - and hipGraphInstantiateFlagAutoFreeOnLaunch for following scenario :
 * - 1) Take three host arrays (hostMem1, hostMem2, hostMem3)
 * - 2) Create one graph to copy from hostMem1 to hostMem2 - hosmemcpyNodeH2H
 * - 3) Create graphExec1 with flag 0
 * - 4) Create another graph with following nodes,
 * -     a. Node to allocate memory - memAllocNode
 * -     b. Node to copy from hostMem2 to device - memcpyNodeH2D
 * -     c. Node to perform double operation - kernelNode
 * -     d. Node to copy from device to host - memcpyNodeD2H
 * - 5) Create graphExec2 with flag hipGraphInstantiateFlagAutoFreeOnLaunch
 * - 6) Launch the graph1, graph2 in a loop and for each iteration
 * -    fill new value in hostMem1.
 * - 7) Each time hostMem3 should contain the expected value
 * -    and it should not give memory related issues.
 * Test source
 * ------------------------
 * - unit/graph/hipGraphInstantiateWithFlags.cc
 */
TEST_CASE("Unit_hipGraphInstantiateWithFlags_WithDefaultAndAutoFreeOnLaunch") {
  int* hostMem1 = new int[SIZE];
  REQUIRE(hostMem1 != nullptr);
  int* hostMem2 = new int[SIZE];
  REQUIRE(hostMem2 != nullptr);
  int* hostMem3 = new int[SIZE];
  REQUIRE(hostMem3 != nullptr);
  int* devMem = nullptr;

  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  hipGraph_t graph1, graph2;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  // Prepare graph1, graphExec1
  hipGraphNode_t memcpyNodeH2H;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2H, graph1, nullptr, 0, hostMem2, hostMem1, NBYTES,
                                    hipMemcpyHostToHost));

  hipGraphExec_t graphExec1;
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec1, graph1, 0));

  // Prepare graph2, graphExec2
  hipGraphNode_t memAllocNode, memcpyNodeH2D, kernelNode, memcpyNodeD2H;
  hipMemAllocNodeParams memAllocNodeParams{};
  memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
  memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
  memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
  memAllocNodeParams.poolProps.location.id = 0;
  memAllocNodeParams.bytesize = NBYTES;

  HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode, graph2, nullptr, 0, &memAllocNodeParams));
  devMem = reinterpret_cast<int*>(memAllocNodeParams.dptr);
  REQUIRE(devMem != nullptr);

  ::std::vector<hipGraphNode_t> memcpyNodeH2DDependencies;
  memcpyNodeH2DDependencies.push_back(memAllocNode);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph2, memcpyNodeH2DDependencies.data(),
                                    memcpyNodeH2DDependencies.size(), devMem, hostMem2, NBYTES,
                                    hipMemcpyHostToDevice));

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(doubleKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  int size = SIZE;
  void* kernelArgs[2] = {reinterpret_cast<void*>(&devMem), reinterpret_cast<void*>(&size)};
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph2, kernelNodeDependencies.data(),
                                  kernelNodeDependencies.size(), &kernelNodeParams));

  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph2, memcpyNodeD2HDependencies.data(),
                                    memcpyNodeD2HDependencies.size(), hostMem3, devMem, NBYTES,
                                    hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec2;
  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec2, graph2, hipGraphInstantiateFlagAutoFreeOnLaunch));

  for (int launch = 1; launch <= 10; launch++) {
    std::fill(hostMem1, hostMem1 + SIZE, launch);
    std::fill(hostMem2, hostMem2 + SIZE, 0);
    std::fill(hostMem3, hostMem3 + SIZE, 0);

    HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));

    HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
    HIP_CHECK(hipStreamSynchronize(stream2));

    for (int idx = 0; idx < SIZE; idx++) {
      INFO("For Launch : " << launch << ", At index : " << idx << ", Got value : " << hostMem3[idx]
                           << ", Expected value : " << (launch + launch) << "\n");
      REQUIRE(hostMem3[idx] == (launch + launch));
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  delete[] hostMem1;
  delete[] hostMem2;
  delete[] hostMem3;
  HIP_CHECK(hipFree(devMem));
}
