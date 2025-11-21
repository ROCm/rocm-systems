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
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>


#define CODEOBJ_FILE "add_Kernel.code"
#define KERNEL_NAME "Add"
#define THREADS_PER_BLOCK 512

/**
* @addtogroup hipGraphAddKernelNode hipModuleLoad hipModuleGetFunction
* @{
* @ingroup GraphTest
* `hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    const hipKernelNodeParams* pNodeParams)` -
* Creates a kernel execution node and adds it to a graph
* `hipError_t hipModuleLoad(hipModule_t* module, const char* fname)` -
* Loads code object from file into a module the currrent context
* `hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname)`
-
* Function with kname will be extracted if present in module
*/

/**
 * Test Description
 * ------------------------
 * - Test case to verify negative scenarios of hipGraphAddKernelNode API.
 * Test source
 * ------------------------
 * - catch/unit/graph/hipGraphAddKernelNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

constexpr size_t size = 1 << 12;
enum fnType { normal, object };

TEST_CASE("Unit_hipGraphAddKernelNode_Negative") {
  constexpr int N = 1024;
  size_t NElem{N};
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  int *A_d, *B_d, *C_d;
  hipGraph_t graph;
  hipGraphNode_t kNode;
  hipKernelNodeParams kNodeParams{};
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipMalloc(&A_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&B_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&C_d, sizeof(int) * N));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);

  SECTION("Pass pGraphNode as nullptr") {
    HIP_CHECK_ERROR(hipGraphAddKernelNode(nullptr, graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass Graph as nullptr") {
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, nullptr, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies") {
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, nullptr, 11, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies and valid list for dependencies") {
    HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));
    dependencies.push_back(kNode);
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, dependencies.data(),
                                          dependencies.size() + 1, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass NodeParams as nullptr") {
    HIP_CHECK_ERROR(
        hipGraphAddKernelNode(&kNode, graph, dependencies.data(), dependencies.size(), nullptr),
        hipErrorInvalidValue);
  }

#if HT_NVIDIA  // on AMD this returns hipErrorInvalidValue
  SECTION("Pass NodeParams func data member as nullptr") {
    kNodeParams.func = nullptr;
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidDeviceFunction);
  }
#endif

  SECTION("Pass kernelParams data member as nullptr") {
    kNodeParams.kernelParams = nullptr;
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }

#if HT_AMD  // On Cuda setup this test case getting failed
  SECTION("Try adding kernel node after destroy the already created graph") {
    hipGraph_t destroyed_graph;
    HIP_CHECK(hipGraphCreate(&destroyed_graph, 0));
    HIP_CHECK(hipGraphDestroy(destroyed_graph));
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, destroyed_graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipGraphDestroy(graph));
}
#if HT_AMD
static __global__ void Add(int* A_d, int* B_d, int* C_d) {
  size_t tx = (blockIdx.x * blockDim.x + threadIdx.x);
  C_d[tx] = A_d[tx] + B_d[tx];
}
static void validateOutput(const hipGraph_t& graph, int* A_h, int* B_h, int* C_h,
                           size_t inputSize) {
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  // Verify output
  for (size_t i = 0; i < inputSize; i++) {
    REQUIRE((A_h[i] + B_h[i]) == C_h[i]);
  }
}
static void kernelFnChange(int* A_d, int* A_h, int* B_d, int* B_h, int* C_d, int* C_h,
                           size_t inputSize, size_t numOfBlocks, enum fnType fn) {
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memcpyNode, memcpyNode1, memcpyNode2, kernelNode;

  hipModule_t Module;
  hipFunction_t Function;
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIPCHECK(hipModuleGetFunction(&Function, Module, KERNEL_NAME));

  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Add MemCpy nodes H2D
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h,
                                    sizeof(int) * inputSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode1, graph, nullptr, 0, B_d, B_h,
                                    sizeof(int) * inputSize, hipMemcpyHostToDevice));
  nodeDependencies.push_back(memcpyNode);
  nodeDependencies.push_back(memcpyNode1);
  // kernel node.
  hipKernelNodeParams kernelNodeParams{}, kernelNodeParamsUpdate{};
  void* kernelArgs[4] = {reinterpret_cast<void*>(&A_d), reinterpret_cast<void*>(&B_d),
                         reinterpret_cast<void*>(&C_d), &numOfBlocks};
  if (fn == normal) {  // normal function
    kernelNodeParams.func = reinterpret_cast<void*>(Add);
  } else {  // Code Object function
    kernelNodeParams.func = reinterpret_cast<void*>(Function);
  }
  kernelNodeParams.gridDim = dim3(inputSize / THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &kernelNodeParams));
  if (fn == normal) {
    kernelNodeParamsUpdate.func = reinterpret_cast<void*>(Function);
  } else {
    kernelNodeParamsUpdate.func = reinterpret_cast<void*>(Add);
  }
  kernelNodeParamsUpdate.gridDim = dim3(inputSize / THREADS_PER_BLOCK, 1, 1);
  kernelNodeParamsUpdate.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParamsUpdate.sharedMemBytes = 0;
  kernelNodeParamsUpdate.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParamsUpdate.extra = nullptr;
  HIP_CHECK(hipGraphKernelNodeSetParams(kernelNode, &kernelNodeParamsUpdate));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  // Add MemCpy nodes D2H
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode2, graph, nodeDependencies.data(),
                                    nodeDependencies.size(), C_h, C_d, sizeof(int) * inputSize,
                                    hipMemcpyDeviceToHost));
  nodeDependencies.clear();

  // Validation
  validateOutput(graph, A_h, B_h, C_h, inputSize);

  HIP_CHECK(hipGraphDestroy(graph));
  HIPCHECK(hipModuleUnload(Module));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify kernel function output in the graph, cloned graph by adding
 *   hipGraphAddKernelNode by loading kernerl function through hipModuleLoad,
 *   hipModuleGetFunction from the code object file.
 * Test source
 * ------------------------
 * - catch/unit/graph/hipGraphAddKernelNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipGraphAddKernelNode_moduleLoadKernelFn_graphNclonedGraph") {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, size, false);

  hipGraph_t graph, clonedGraph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memcpyNode, memcpyNode1, memcpyNode2, kernelNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipModule_t Module;
  hipFunction_t Function;
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIPCHECK(hipModuleGetFunction(&Function, Module, KERNEL_NAME));

  // Add MemCpy nodes H2D
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h, sizeof(int) * size,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode1, graph, nullptr, 0, B_d, B_h, sizeof(int) * size,
                                    hipMemcpyHostToDevice));
  nodeDependencies.push_back(memcpyNode);
  nodeDependencies.push_back(memcpyNode1);

  // Add Kernel Node
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[3] = {reinterpret_cast<void*>(&A_d), reinterpret_cast<void*>(&B_d),
                         reinterpret_cast<void*>(&C_d)};
  kernelNodeParams.func = reinterpret_cast<void*>(Function);
  kernelNodeParams.gridDim = dim3(size / THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  // Add MemCpy nodes D2H
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode2, graph, nodeDependencies.data(),
                                    nodeDependencies.size(), C_h, C_d, sizeof(int) * size,
                                    hipMemcpyDeviceToHost));
  nodeDependencies.clear();
  SECTION("Original Graph") {
    // Original Graph validation
    validateOutput(graph, A_h, B_h, C_h, size);
  }
  SECTION("Cloned Graph") {
    // Clone the graph
    HIP_CHECK(hipGraphClone(&clonedGraph, graph));
    // Cloned graph Validation
    validateOutput(clonedGraph, A_h, B_h, C_h, size);
    HIP_CHECK(hipGraphDestroy(clonedGraph));
  }
  HIP_CHECK(hipGraphDestroy(graph));
  HIPCHECK(hipModuleUnload(Module));
  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify kernel function output by adding hipGraphAddKernelNode and updating the
 *   kernel functions from normal to Code object and vice versa in the graph by loading kernerl
 *   function through hipModuleLoad, hipModuleGetFunction from code object file.
 * Test source
 * ------------------------
 * - catch/unit/graph/hipGraphAddKernelNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipGraphAddKernelNode_moduleLoadKernelFn_kernelFnUpdate") {
  size_t maxBlocks = 512;
  int *A_d, *B_d, *C_d;  // Device pointers
  int *A_h, *B_h, *C_h;  // Host Pointers
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, size, false);

  SECTION("Kernel function change from Normal fn to Code object fn") {
    kernelFnChange(A_d, A_h, B_d, B_h, C_d, C_h, size, maxBlocks, object);
  }
  SECTION("Kernel function change from Code object fn to normal fn") {
    kernelFnChange(A_d, A_h, B_d, B_h, C_d, C_h, size, maxBlocks, normal);
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify kernel function output in the child graph and cloned graph by adding
 *   hipGraphAddKernelNode by loading kernerl function through hipModuleLoad,
 *   hipModuleGetFunction from the code object file.
 * Test source
 * ------------------------
 * - catch/unit/graph/hipGraphAddKernelNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipGraphAddKernelNode_moduleLoadKernelFn_childGraph") {
  int *A_d, *B_d, *C_d;  // Device pointers
  int *A_h, *B_h, *C_h;  // Host Pointers
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, size, false);

  hipGraph_t graph, childgraph, clonedGraph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memcpyh2d1, memcpyh2d2, memcpyd2h, childGraphNode, kernelNode;

  hipModule_t Module;
  hipFunction_t Function;
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIPCHECK(hipModuleGetFunction(&Function, Module, KERNEL_NAME));

  // Create child graph
  HIP_CHECK(hipGraphCreate(&childgraph, 0));

  // kerrel params.
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[3] = {reinterpret_cast<void*>(&A_d), reinterpret_cast<void*>(&B_d),
                         reinterpret_cast<void*>(&C_d)};
  kernelNodeParams.func = reinterpret_cast<void*>(Function);
  kernelNodeParams.gridDim = dim3(size / THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, childgraph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Add MemCpy nodes H2D
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyh2d1, graph, nullptr, 0, A_d, A_h, sizeof(int) * size,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyh2d2, graph, nullptr, 0, B_d, B_h, sizeof(int) * size,
                                    hipMemcpyHostToDevice));
  nodeDependencies.push_back(memcpyh2d1);
  nodeDependencies.push_back(memcpyh2d2);
  // Add child graph node
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nodeDependencies.data(),
                                      nodeDependencies.size(), childgraph));
  nodeDependencies.clear();
  nodeDependencies.push_back(childGraphNode);

  // Add MemCpy nodes D2H
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyd2h, graph, nodeDependencies.data(),
                                    nodeDependencies.size(), C_h, C_d, sizeof(int) * size,
                                    hipMemcpyDeviceToHost));
  nodeDependencies.clear();

  SECTION("Original Graph") {
    // Original Graph validation
    validateOutput(graph, A_h, B_h, C_h, size);
  }
  SECTION("Cloned Graph") {
    // Clone the graph
    HIP_CHECK(hipGraphClone(&clonedGraph, graph));
    // Cloned Graph validation
    validateOutput(clonedGraph, A_h, B_h, C_h, size);
    HIP_CHECK(hipGraphDestroy(clonedGraph));
  }
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIPCHECK(hipModuleUnload(Module));
  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify kernel function output in the graph which is created by stream capture.
 *   The kernel function is loading through hipModuleLoad, hipModuleGetFunction from code object
 * file Test source
 * ------------------------
 * - catch/unit/graph/hipGraphAddKernelNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipGraphAddKernelNode_moduleLoadKernelFn_streamCapture") {
  size_t maxBlocks = 512;
  size_t Nbytes = sizeof(int) * maxBlocks;

  int *A_d, *B_d, *C_d;  // Device pointers
  int *A_h, *B_h, *C_h;  // Host Pointers
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, maxBlocks, false);

  hipGraph_t graph;
  hipStream_t stream;

  hipModule_t Module;
  hipFunction_t Function;
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIPCHECK(hipModuleGetFunction(&Function, Module, KERNEL_NAME));

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  // MemCpy node H2D
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  // kerrel params.
  void* kernelArgs[] = {&A_d, &B_d, &C_d};

  // Kernel node
  HIP_CHECK(
      hipModuleLaunchKernel(Function, 1, 1, 1, maxBlocks, 1, 1, 0, stream, kernelArgs, nullptr));

  // MemCpy nodes D2H
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipStreamDestroy(stream));

  // validation
  validateOutput(graph, A_h, B_h, C_h, maxBlocks);

  HIP_CHECK(hipGraphDestroy(graph));
  HIPCHECK(hipModuleUnload(Module));
  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
}
#endif


/**
 * End doxygen group GraphTest.
 * @}
 */
