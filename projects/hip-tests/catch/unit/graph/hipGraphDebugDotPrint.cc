/*
Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * @addtogroup hipGraphDebugDotPrint hipGraphDebugDotPrint
 * @{
 * @ingroup GraphTest
 * `hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags)` -
 * Write a DOT file describing graph structure.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define N 1024

#ifdef __linux__
#include <unistd.h>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>

__device__ int globalIn[N];

static void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  std::iota(A, A + N, 0);
}

static void deleteFile(const char* fName) {
  if (remove(fName) != 0) {
    INFO("Error in deleting file -" << fName);
  } else {
    INFO("Successfully deleted file -" << fName);
  }
}

static bool checkFileExists(const char* fName) { return (access(fName, F_OK) != -1); }

static unsigned countSubstr(const std::string& input_str, const std::string& substr) {
  unsigned count = 0;
  std::string::size_type srch_pos = 0, cur_pos = 0;
  while ((cur_pos = input_str.find(substr, srch_pos)) != std::string::npos) {
    count++;
    srch_pos = (cur_pos + substr.length());
  }
  return count;
}

static bool validateDotFile(const char* fName, const std::map<std::string, unsigned>& graphData) {
  std::ifstream infile(fName);
  std::stringstream buffer;
  buffer << infile.rdbuf();
  const std::string buffer_str = buffer.str();
  for (auto it = graphData.begin(); it != graphData.end(); it++) {
    unsigned count = countSubstr(buffer_str, it->first);
    if (it->second != count) {
      INFO("validateDotFile: Failed for key :: " << it->first << " : " << count
                                                 << " Expected : " << it->second);
      return false;
    }
  }
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hipGraphDebugDotPrint
 *   Call hipGraphDebugDotPrint and provice path where to write the DOT file.
 *   Verify that DOT file get created or not for each flag passed.
 *   1) Add MemcpyNode node to graph & validate its DebugDotPrint descriptions
 *   2) Add kernel node to graph & validate its DebugDotPrint descriptions
 *   3) Add memset node to graph & validate its DebugDotPrint descriptions
 *   4) Add emptyNode to graph & validate its DebugDotPrint descriptions
 *   5) Add childGraphNode to graph & validate its DebugDotPrint descriptions
 *   6) Add eventRecord to graph & validate its DebugDotPrint descriptions
 *   7) Add eventWait to graph & validate its DebugDotPrint descriptions
 *   8) Add hostNode to graph & validate its DebugDotPrint descriptions
 *   9) Add mecpyNode1D to graph & validate its DebugDotPrint descriptions
 *   10) Add mecpyNode3D to graph & validate its DebugDotPrint descriptions
 *   11) Add MemcpyNodeToSymbol to graph & validate its DebugDotPrint descriptions
 *   12) Add MemcpyNodeFromSymbol to graph & validate its DebugDotPrint descriptions
 *   13) Add Dependencies to graph & validate its DebugDotPrint descriptions
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphDebugDotPrint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

static void hipGraphDebugDotPrint_Functional(const char* fName, unsigned int flag = 0) {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, childGraph;
  hipStream_t stream;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kNodeAdd, memsetNode;
  hipGraphNode_t emptyNode, childGraphNode, eventWait, eventRecord, hostNode;
  hipKernelNodeParams kNodeParams{};
  int *A_d, *B_d, *C_d, *mem_d;
  int *A_h, *B_h, *C_h, *mem_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  mem_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(hipMalloc(&mem_d, Nbytes));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;

  // Add Kernel node to graph & validate its DebugDotPrint descriptions
  HIP_CHECK(hipGraphAddKernelNode(&kNodeAdd, graph, nullptr, 0, &kNodeParams));

  // Add MemCpy node to graph & validate its DebugDotPrint descriptions
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  // Add Dependencies to graph & validate its DebugDotPrint descriptions
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kNodeAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kNodeAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNodeAdd, &memcpy_C, 1));

  // Add emptyNode to graph & validate its DebugDotPrint descriptions
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

  // Add hostNode to graph & validate its DebugDotPrint descriptions
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = mem_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  // Add eventRecord to graph & validate its DebugDotPrint descriptions
  HIP_CHECK(hipGraphAddEventRecordNode(&eventRecord, graph, nullptr, 0, event));

  // Add eventWait to graph & validate its DebugDotPrint descriptions
  HIP_CHECK(hipGraphAddEventWaitNode(&eventWait, graph, nullptr, 0, event));

  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  // Add emcpyNode3D to graph & validate its DebugDotPrint descriptions
  constexpr int width{10}, height{10}, depth{10};
  hipArray_t devArray1;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparams;
  uint32_t size = width * height * depth * sizeof(int);
  hipGraphNode_t mcpyNode3D;
  int* hData = reinterpret_cast<int*>(malloc(size));
  REQUIRE(hData != nullptr);

  // Initialize host buffer
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        hData[i * width * height + j * width + k] = i * width * height + j * width + k;
      }
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int) * 8, 0, 0, 0, formatKind);
  HIP_CHECK(hipMalloc3DArray(&devArray1, &channelDesc, make_hipExtent(width, height, depth),
                             hipArrayDefault));

  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, height, depth);
  myparams.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int), width, height);
  myparams.dstArray = devArray1;
  myparams.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipGraphAddMemcpyNode(&mcpyNode3D, childGraph, nullptr, 0, &myparams));

  // Add MemcpyNodeToSymbol to graph & validate its DebugDotPrint description
  hipGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode;

  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, childGraph, nullptr, 0,
                                          HIP_SYMBOL(globalIn), B_h, Nbytes, 0,
                                          hipMemcpyHostToDevice));

  // Add MemcpyNodeFromSymbol to graph & validate its DebugDotPrint description
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, childGraph, nullptr, 0, B_h,
                                            HIP_SYMBOL(globalIn), Nbytes, 0,
                                            hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(childGraph, &memcpyToSymbolNode, &memcpyFromSymbolNode, 1));
  // Add memset node to graph & validate its DebugDotPrint descriptions
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(mem_d);
  memsetParams.value = 7;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, childGraph, nullptr, 0, &memsetParams));

  // Add childGraphNode to graph & validate its DebugDotPrint descriptions
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));

  std::map<std::string, unsigned> graphData;
  graphData["->"] = 4;  //  number of edges
  graphData["MEMCPY"] = 6;
  graphData["HtoA"] = 1;
  graphData["HtoD"] = 3;
  graphData["DtoH"] = 2;
  graphData["MEMSET"] = 1;
  graphData["EMPTY"] = 1;
  graphData["EVENT_WAIT"] = 1;
  graphData["EVENT_RECORD"] = 1;
  graphData["subgraph"] = 2;
  graphData["HOST"] = 1;

#if HT_NVIDIA
  if (flag == hipGraphDebugDotFlagsVerbose || flag == hipGraphDebugDotFlagsMemcpyNodeParams) {
    graphData["HOST"] = 7;
  }
#endif

  if (flag == hipGraphDebugDotFlagsVerbose || flag == hipGraphDebugDotFlagsKernelNodeAttributes) {
    graphData["KERNEL"] = 1;
  }

  HIP_CHECK(hipGraphDebugDotPrint(graph, fName, flag));
  REQUIRE(true == checkFileExists(fName));
  REQUIRE(true == validateDotFile(fName, graphData));
  deleteFile(fName);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  free(mem_h);
  HIP_CHECK(hipFree(mem_d));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(childGraph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/* Functional Test for API - hipGraphDebugDotPrint
   Call hipGraphDebugDotPrint and provice path where to write the DOT file.
   Verify that DOT file get created or not for each flag passed. */

TEST_CASE("Unit_hipGraphDebugDotPrint_Functional") {
  CHECK_IMAGE_SUPPORT

  SECTION("Call with hipGraphDebugDotFlagsVerbose flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncVerbose.dot", hipGraphDebugDotFlagsVerbose);
  }
  SECTION("Call with hipGraphDebugDotFlagsKernelNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncKernelParams.dot",
                                     hipGraphDebugDotFlagsKernelNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsMemcpyNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncMemcpy.dot",
                                     hipGraphDebugDotFlagsMemcpyNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsMemsetNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncMemset.dot",
                                     hipGraphDebugDotFlagsMemsetNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsHostNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncHost.dot",
                                     hipGraphDebugDotFlagsHostNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsEventNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncEvent.dot",
                                     hipGraphDebugDotFlagsEventNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsExtSemasSignalNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncExtSemasSignal.dot",
                                     hipGraphDebugDotFlagsExtSemasSignalNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsExtSemasWaitNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncExtSemasWait.dot",
                                     hipGraphDebugDotFlagsExtSemasWaitNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsKernelNodeAttributes flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncKernelNodeAttr.dot",
                                     hipGraphDebugDotFlagsKernelNodeAttributes);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative Test for API - hipGraphDebugDotPrint Argument Check
 *   1) Pass graph as nullptr
 *   2) Pass graph as uninitialize structure
 *   3) Pass path for dot file to store as nullptr
 *   4) Pass path for dot file to store as empth path
 *   5) Pass flag as hipGraphDebugDotFlags MIN - 1
 *   6) Pass flag as hipGraphDebugDotFlags MAX + 1
 *   7) Pass flag as INT_MAX
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphDebugDotPrint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

#define DOT_FILE_PATH_NEG "./graphDotFileNeg.dot"

TEST_CASE("Unit_hipGraphDebugDotPrint_Argument_Check") {
  hipGraph_t graph;
  hipError_t ret;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  SECTION("Pass graph as nullptr") {
    ret = hipGraphDebugDotPrint(nullptr, DOT_FILE_PATH_NEG, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass graph as uninitialize structure") {
    hipGraph_t graphT{};
    ret = hipGraphDebugDotPrint(graphT, DOT_FILE_PATH_NEG, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass path for dot file to store as nullptr") {
    ret = hipGraphDebugDotPrint(graph, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass path for dot file to store as empth path") {
    ret = hipGraphDebugDotPrint(graph, "", 0);
    REQUIRE(hipErrorOperatingSystem == ret);
  }
  SECTION("Pass flag as hipGraphDebugDotFlags MIN - 1") {
    ret = hipGraphDebugDotPrint(graph, DOT_FILE_PATH_NEG, hipGraphDebugDotFlagsVerbose - 1);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass flag as hipGraphDebugDotFlags MAX + 1") {
    ret = hipGraphDebugDotPrint(graph, DOT_FILE_PATH_NEG, hipGraphDebugDotFlagsHandles + 1);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass flag as INT_MAX") {
    ret = hipGraphDebugDotPrint(graph, DOT_FILE_PATH_NEG, INT_MAX);
    REQUIRE(hipSuccess == ret);
  }
  deleteFile(DOT_FILE_PATH_NEG);
  HIP_CHECK(hipGraphDestroy(graph));
}
#endif  //  __linux__

/**
 * End doxygen group GraphTest.
 * @}
 */
