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

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_defgroups.hh>
#include <utils.hh>
#include "../device/hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (stream capture) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_StreamCapture") {
  void* hipStreamBeginCapture_ptr = nullptr;
  void* hipStreamIsCapturing_ptr = nullptr;
  void* hipStreamEndCapture_ptr = nullptr;
  void* hipStreamAddCallback_ptr = nullptr;
  void* hipGraphInstantiate_ptr = nullptr;
  void* hipGraphLaunch_ptr = nullptr;
  void* hipGraphExecDestroy_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipStreamBeginCapture", &hipStreamBeginCapture_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamIsCapturing", &hipStreamIsCapturing_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamEndCapture", &hipStreamEndCapture_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamAddCallback", &hipStreamAddCallback_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphInstantiate", &hipGraphInstantiate_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipGraphLaunch", &hipGraphLaunch_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphExecDestroy", &hipGraphExecDestroy_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipStreamBeginCapture_ptr)(hipStream_t, hipStreamCaptureMode) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureMode)>(
          hipStreamBeginCapture_ptr);

  hipError_t (*dyn_hipStreamIsCapturing_ptr)(hipStream_t, hipStreamCaptureStatus*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureStatus*)>(
          hipStreamIsCapturing_ptr);

  hipError_t (*dyn_hipStreamEndCapture_ptr)(hipStream_t, hipGraph_t*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipGraph_t*)>(hipStreamEndCapture_ptr);

  hipError_t (*dyn_hipStreamAddCallback_ptr)(hipStream_t, hipStreamCallback_t, void*,
                                             unsigned int) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCallback_t, void*, unsigned int)>(
          hipStreamAddCallback_ptr);

  hipError_t (*dyn_hipGraphInstantiate_ptr)(hipGraphExec_t*, hipGraph_t, hipGraphNode_t*, char*,
                                            size_t) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t*, hipGraph_t, hipGraphNode_t*, char*, size_t)>(
          hipGraphInstantiate_ptr);

  hipError_t (*dyn_hipGraphLaunch_ptr)(hipGraphExec_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipStream_t)>(hipGraphLaunch_ptr);

  hipError_t (*dyn_hipGraphExecDestroy_ptr)(hipGraphExec_t) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t)>(hipGraphExecDestroy_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Validating hipStreamBeginCapture API
  HIP_CHECK(dyn_hipStreamBeginCapture_ptr(stream, hipStreamCaptureModeGlobal));

  // Validating hipStreamIsCapturing API
  hipStreamCaptureStatus pCaptureStatus = hipStreamCaptureStatusNone;
  HIP_CHECK(dyn_hipStreamIsCapturing_ptr(stream, &pCaptureStatus));
  REQUIRE(pCaptureStatus == hipStreamCaptureStatusActive);

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes, hipMemcpyHostToDevice, stream));
  addOneKernel<<<1, 1, 0, stream>>>(devMem, N);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost, stream));

  // Validating hipStreamEndCapture API
  HIP_CHECK(dyn_hipStreamEndCapture_ptr(stream, &graph));

  // Validating hipStreamAddCallback API
  int data = 100;
  HIP_CHECK(
      dyn_hipStreamAddCallback_ptr(stream, callBackFunction, reinterpret_cast<void*>(&data), 0));

  // Validating hipGraphInstantiate API
  hipGraphExec_t graphExec;
  HIP_CHECK(dyn_hipGraphInstantiate_ptr(&graphExec, graph, nullptr, nullptr, 0));

  // Validating hipGraphLaunch API
  HIP_CHECK(dyn_hipGraphLaunch_ptr(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(validateHostArray(hostMem, N, 11) == true);
  REQUIRE(data == 200);

  // Validating hipGraphExecDestroy API
  HIP_CHECK(dyn_hipGraphExecDestroy_ptr(graphExec));
  REQUIRE(dyn_hipGraphLaunch_ptr(graphExec, stream) == hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (adding memcpy 1D and kernel nodes) APIs from the hipGetProcAddress
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_AddMemcpy1DKernelNodes") {
  void* hipGraphAddMemcpyNode1D_ptr = nullptr;
  void* hipGraphAddKernelNode_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphAddMemcpyNode1D", &hipGraphAddMemcpyNode1D_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphAddKernelNode", &hipGraphAddKernelNode_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddMemcpyNode1D_ptr)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                                                size_t, void*, const void*, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t,
                                      void*, const void*, size_t, hipMemcpyKind)>(
          hipGraphAddMemcpyNode1D_ptr);

  hipError_t (*dyn_hipGraphAddKernelNode_ptr)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                                              size_t, const hipKernelNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t,
                                      const hipKernelNodeParams*)>(hipGraphAddKernelNode_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 100);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t memcpyNodeH2D, kernelNode, memcpyNodeD2H;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Validating hipGraphAddMemcpyNode1D API
  // Prepare memcpyNodeH2D
  HIP_CHECK(dyn_hipGraphAddMemcpyNode1D_ptr(&memcpyNodeH2D, graph, nullptr, 0, devMem, hostMem,
                                            Nbytes, hipMemcpyHostToDevice));

  // Validating hipGraphAddKernelNode API
  // Prepare kernelNode with memcpyNodeH2D as a dependency
  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = {reinterpret_cast<void*>(&devMem), reinterpret_cast<void*>(&N)};
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(dyn_hipGraphAddKernelNode_ptr(&kernelNode, graph, kernelNodeDependencies.data(),
                                          kernelNodeDependencies.size(), &kernelNodeParams));

  // Prepare memcpyNodeD2H with kernelNode as a dependency
  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);
  HIP_CHECK(dyn_hipGraphAddMemcpyNode1D_ptr(&memcpyNodeD2H, graph, memcpyNodeD2HDependencies.data(),
                                            memcpyNodeD2HDependencies.size(), hostMem, devMem,
                                            Nbytes, hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  REQUIRE(validateHostArray(hostMem, N, 101) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (adding memset and memcpy nodes) APIs from the hipGetProcAddress
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_AddMemsetMemcpyNodes") {
  CHECK_IMAGE_SUPPORT

  void* hipGraphAddMemsetNode_ptr = nullptr;
  void* hipGraphAddMemcpyNode_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphAddMemsetNode", &hipGraphAddMemsetNode_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphAddMemcpyNode", &hipGraphAddMemcpyNode_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddMemsetNode_ptr)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                                              size_t, const hipMemsetParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t,
                                      const hipMemsetParams*)>(hipGraphAddMemsetNode_ptr);

  hipError_t (*dyn_hipGraphAddMemcpyNode_ptr)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                                              size_t, const hipMemcpy3DParms*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t,
                                      const hipMemcpy3DParms*)>(hipGraphAddMemcpyNode_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char* devMemSrc = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc), &pitch, width, height));
  REQUIRE(devMemSrc != nullptr);

  char* hostMemDst = reinterpret_cast<char*>(malloc(N * sizeof(char)));
  REQUIRE(hostMemDst != nullptr);

  hipGraphNode_t memsetNode, memcpyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Validating hipGraphAddMemsetNode API
  // Prepare memsetNode
  hipMemsetParams pMemsetParams{};
  pMemsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  pMemsetParams.value = value;
  pMemsetParams.pitch = pitch;
  pMemsetParams.elementSize = sizeof(char);
  pMemsetParams.width = width;
  pMemsetParams.height = height;

  HIP_CHECK(dyn_hipGraphAddMemsetNode_ptr(&memsetNode, graph, nullptr, 0, &pMemsetParams));

  // Validating hipGraphAddMemcpyNode API
  // Prepare memcpyNode with memsetNode as a dependency
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(devMemSrc, pitch, width, height);
  myparms.dstPtr = make_hipPitchedPtr(hostMemDst, width, width, height);
  myparms.extent = make_hipExtent(width, height, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(dyn_hipGraphAddMemcpyNode_ptr(&memcpyNode, graph, memcpyNodeDependencies.data(),
                                          memcpyNodeDependencies.size(), &myparms));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  REQUIRE(validateArrayT<char>(hostMemDst, N, value) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMemSrc));
  free(hostMemDst);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (set/get params for memset and memcpy nodes) APIs from the
 *  - hipGetProcAddress and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_SetGetParamsMemsetMemcpy") {
  CHECK_IMAGE_SUPPORT

  void* hipGraphMemsetNodeSetParams_ptr = nullptr;
  void* hipGraphMemsetNodeGetParams_ptr = nullptr;
  void* hipGraphMemcpyNodeSetParams_ptr = nullptr;
  void* hipGraphMemcpyNodeGetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphMemsetNodeSetParams", &hipGraphMemsetNodeSetParams_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphMemsetNodeGetParams", &hipGraphMemsetNodeGetParams_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphMemcpyNodeSetParams", &hipGraphMemcpyNodeSetParams_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphMemcpyNodeGetParams", &hipGraphMemcpyNodeGetParams_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphMemsetNodeSetParams_ptr)(hipGraphNode_t, const hipMemsetParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, const hipMemsetParams*)>(
          hipGraphMemsetNodeSetParams_ptr);

  hipError_t (*dyn_hipGraphMemsetNodeGetParams_ptr)(hipGraphNode_t, hipMemsetParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipMemsetParams*)>(
          hipGraphMemsetNodeGetParams_ptr);

  hipError_t (*dyn_hipGraphMemcpyNodeSetParams_ptr)(hipGraphNode_t, const hipMemcpy3DParms*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, const hipMemcpy3DParms*)>(
          hipGraphMemcpyNodeSetParams_ptr);

  hipError_t (*dyn_hipGraphMemcpyNodeGetParams_ptr)(hipGraphNode_t, hipMemcpy3DParms*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipMemcpy3DParms*)>(
          hipGraphMemcpyNodeGetParams_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char* devMemSrc1 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc1), &pitch, width, height));
  REQUIRE(devMemSrc1 != nullptr);

  char* hostMemDst1 = reinterpret_cast<char*>(malloc(N * sizeof(char)));
  REQUIRE(hostMemDst1 != nullptr);
  fillCharHostArray(hostMemDst1, N, 100);

  char* devMemSrc2 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc2), &pitch, width, height));
  REQUIRE(devMemSrc2 != nullptr);

  char* hostMemDst2 = reinterpret_cast<char*>(malloc(N * sizeof(char)));
  REQUIRE(hostMemDst2 != nullptr);
  fillCharHostArray(hostMemDst2, N, 100);

  hipGraphNode_t memsetNode, memcpyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memset node
  hipMemsetParams initialMemsetParams{};
  initialMemsetParams.dst = reinterpret_cast<void*>(devMemSrc1);
  initialMemsetParams.value = value;
  initialMemsetParams.pitch = pitch;
  initialMemsetParams.elementSize = sizeof(char);
  initialMemsetParams.width = width;
  initialMemsetParams.height = height;

  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &initialMemsetParams));

  hipMemsetParams receivedMemsetValues{};
  HIP_CHECK(dyn_hipGraphMemsetNodeGetParams_ptr(memsetNode, &receivedMemsetValues));

  REQUIRE(receivedMemsetValues.dst == devMemSrc1);
  REQUIRE(receivedMemsetValues.value == value);
  REQUIRE(receivedMemsetValues.pitch == pitch);
  REQUIRE(receivedMemsetValues.elementSize == sizeof(char));
  REQUIRE(receivedMemsetValues.width == width);
  REQUIRE(receivedMemsetValues.height == height);

  hipMemsetParams correctedMemsetParams{};
  correctedMemsetParams.dst = reinterpret_cast<void*>(devMemSrc2);
  correctedMemsetParams.value = value;
  correctedMemsetParams.pitch = pitch;
  correctedMemsetParams.elementSize = sizeof(char);
  correctedMemsetParams.width = width;
  correctedMemsetParams.height = height;

  // Validating hipGraphMemsetNodeSetParams API
  HIP_CHECK(dyn_hipGraphMemsetNodeSetParams_ptr(memsetNode, &correctedMemsetParams));

  // Validating hipGraphMemsetNodeGetParams API
  HIP_CHECK(dyn_hipGraphMemsetNodeGetParams_ptr(memsetNode, &receivedMemsetValues));

  REQUIRE(receivedMemsetValues.dst == devMemSrc2);
  REQUIRE(receivedMemsetValues.value == value);
  REQUIRE(receivedMemsetValues.pitch == pitch);
  REQUIRE(receivedMemsetValues.elementSize == sizeof(char));
  REQUIRE(receivedMemsetValues.width == width);
  REQUIRE(receivedMemsetValues.height == height);

  // Prepare memcpyNode
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  hipMemcpy3DParms initialParms{};
  initialParms.srcPos = make_hipPos(0, 0, 0);
  initialParms.dstPos = make_hipPos(0, 0, 0);
  initialParms.srcPtr = make_hipPitchedPtr(devMemSrc1, pitch, width, height);
  initialParms.dstPtr = make_hipPitchedPtr(hostMemDst1, width, width, height);
  initialParms.extent = make_hipExtent(width, height, 1);
  initialParms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, memcpyNodeDependencies.data(),
                                  memcpyNodeDependencies.size(), &initialParms));

  hipMemcpy3DParms receivedMemcpyValues{};
  HIP_CHECK(dyn_hipGraphMemcpyNodeGetParams_ptr(memcpyNode, &receivedMemcpyValues));

  REQUIRE(receivedMemcpyValues.srcArray == nullptr);
  REQUIRE(receivedMemcpyValues.srcPos.x == 0);
  REQUIRE(receivedMemcpyValues.srcPos.y == 0);
  REQUIRE(receivedMemcpyValues.srcPos.z == 0);
  REQUIRE(receivedMemcpyValues.srcPtr.ptr == devMemSrc1);
  REQUIRE(receivedMemcpyValues.srcPtr.pitch == pitch);
  REQUIRE(receivedMemcpyValues.srcPtr.xsize == width);
  REQUIRE(receivedMemcpyValues.srcPtr.ysize == height);
  REQUIRE(receivedMemcpyValues.dstArray == nullptr);
  REQUIRE(receivedMemcpyValues.dstPos.x == 0);
  REQUIRE(receivedMemcpyValues.dstPos.y == 0);
  REQUIRE(receivedMemcpyValues.dstPos.z == 0);
  REQUIRE(receivedMemcpyValues.dstPtr.ptr == hostMemDst1);
  REQUIRE(receivedMemcpyValues.dstPtr.pitch == pitch);
  REQUIRE(receivedMemcpyValues.dstPtr.xsize == width);
  REQUIRE(receivedMemcpyValues.dstPtr.ysize == height);
  REQUIRE(receivedMemcpyValues.extent.width == width);
  REQUIRE(receivedMemcpyValues.extent.height == height);
  REQUIRE(receivedMemcpyValues.extent.depth == 1);
  REQUIRE(receivedMemcpyValues.kind == hipMemcpyDeviceToHost);

  hipMemcpy3DParms correctedParms{};
  correctedParms.srcPos = make_hipPos(0, 0, 0);
  correctedParms.dstPos = make_hipPos(0, 0, 0);
  correctedParms.srcPtr = make_hipPitchedPtr(devMemSrc2, pitch, width, height);
  correctedParms.dstPtr = make_hipPitchedPtr(hostMemDst2, width, width, height);
  correctedParms.extent = make_hipExtent(width, height, 1);
  correctedParms.kind = hipMemcpyDeviceToHost;

  // Validating hipGraphMemcpyNodeSetParams API
  HIP_CHECK(dyn_hipGraphMemcpyNodeSetParams_ptr(memcpyNode, &correctedParms));

  // Validating hipGraphMemcpyNodeGetParams API
  HIP_CHECK(dyn_hipGraphMemcpyNodeGetParams_ptr(memcpyNode, &receivedMemcpyValues));

  REQUIRE(receivedMemcpyValues.srcArray == nullptr);
  REQUIRE(receivedMemcpyValues.srcPos.x == 0);
  REQUIRE(receivedMemcpyValues.srcPos.y == 0);
  REQUIRE(receivedMemcpyValues.srcPos.z == 0);
  REQUIRE(receivedMemcpyValues.srcPtr.ptr == devMemSrc2);
  REQUIRE(receivedMemcpyValues.srcPtr.pitch == pitch);
  REQUIRE(receivedMemcpyValues.srcPtr.xsize == width);
  REQUIRE(receivedMemcpyValues.srcPtr.ysize == height);
  REQUIRE(receivedMemcpyValues.dstArray == nullptr);
  REQUIRE(receivedMemcpyValues.dstPos.x == 0);
  REQUIRE(receivedMemcpyValues.dstPos.y == 0);
  REQUIRE(receivedMemcpyValues.dstPos.z == 0);
  REQUIRE(receivedMemcpyValues.dstPtr.ptr == hostMemDst2);
  REQUIRE(receivedMemcpyValues.dstPtr.pitch == pitch);
  REQUIRE(receivedMemcpyValues.dstPtr.xsize == width);
  REQUIRE(receivedMemcpyValues.dstPtr.ysize == height);
  REQUIRE(receivedMemcpyValues.extent.width == width);
  REQUIRE(receivedMemcpyValues.extent.height == height);
  REQUIRE(receivedMemcpyValues.extent.depth == 1);
  REQUIRE(receivedMemcpyValues.kind == hipMemcpyDeviceToHost);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  REQUIRE(validateArrayT<char>(hostMemDst1, N, 100) == true);
  REQUIRE(validateArrayT<char>(hostMemDst2, N, 120) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMemSrc1));
  HIP_CHECK(hipFree(devMemSrc2));
  free(hostMemDst1);
  free(hostMemDst2);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (set/get params for kernel) APIs from the
 *  - hipGetProcAddress and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_KernelNodeSetGetParams") {
  void* hipGraphKernelNodeGetParams_ptr = nullptr;
  void* hipGraphKernelNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphKernelNodeGetParams", &hipGraphKernelNodeGetParams_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphKernelNodeSetParams", &hipGraphKernelNodeSetParams_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphKernelNodeGetParams_ptr)(hipGraphNode_t, hipKernelNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipKernelNodeParams*)>(
          hipGraphKernelNodeGetParams_ptr);

  hipError_t (*dyn_hipGraphKernelNodeSetParams_ptr)(hipGraphNode_t, const hipKernelNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, const hipKernelNodeParams*)>(
          hipGraphKernelNodeSetParams_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 100);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t memcpyNodeH2D, kernelNode, memcpyNodeD2H;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memcpyNodeH2D
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0, devMem, hostMem, Nbytes,
                                    hipMemcpyHostToDevice));

  // Prepare kernelNode with memcpyNodeH2D as a dependency
  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  void* kernelArgs[2] = {reinterpret_cast<void*>(&devMem), reinterpret_cast<void*>(&N)};

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, kernelNodeDependencies.data(),
                                  kernelNodeDependencies.size(), &kernelNodeParams));

  // Prepare memcpyNodeD2H with kernelNode as a dependency
  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph, memcpyNodeD2HDependencies.data(),
                                    memcpyNodeD2HDependencies.size(), hostMem, devMem, Nbytes,
                                    hipMemcpyDeviceToHost));

  // Get and set Kernel node params
  hipKernelNodeParams receivedKernelNodeParams;

  // Validating hipGraphKernelNodeGetParams API
  HIP_CHECK(dyn_hipGraphKernelNodeGetParams_ptr(kernelNode, &receivedKernelNodeParams));

  REQUIRE(receivedKernelNodeParams.func == addOneKernel);
  REQUIRE(receivedKernelNodeParams.gridDim.x == 1);
  REQUIRE(receivedKernelNodeParams.gridDim.y == 1);
  REQUIRE(receivedKernelNodeParams.gridDim.z == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.x == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.y == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.z == 1);
  REQUIRE(*(reinterpret_cast<int*>(receivedKernelNodeParams.kernelParams[0])) ==
          *(reinterpret_cast<int*>(kernelArgs[0])));
  REQUIRE(*(reinterpret_cast<int*>(receivedKernelNodeParams.kernelParams[1])) ==
          *(reinterpret_cast<int*>(kernelArgs[1])));
  REQUIRE(receivedKernelNodeParams.extra == nullptr);
  REQUIRE(receivedKernelNodeParams.sharedMemBytes == 0);

  hipKernelNodeParams correctedKernelNodeParams{};
  correctedKernelNodeParams.func = reinterpret_cast<void*>(addTwoKernel);
  correctedKernelNodeParams.gridDim = dim3(2, 1, 1);
  correctedKernelNodeParams.blockDim = dim3(2, 1, 1);
  correctedKernelNodeParams.sharedMemBytes = 0;
  correctedKernelNodeParams.kernelParams = kernelArgs;
  correctedKernelNodeParams.extra = nullptr;

  // Validating hipGraphKernelNodeSetParams API
  HIP_CHECK(dyn_hipGraphKernelNodeSetParams_ptr(kernelNode, &correctedKernelNodeParams));

  HIP_CHECK(dyn_hipGraphKernelNodeGetParams_ptr(kernelNode, &receivedKernelNodeParams));

  REQUIRE(receivedKernelNodeParams.func == addTwoKernel);
  REQUIRE(receivedKernelNodeParams.gridDim.x == 2);
  REQUIRE(receivedKernelNodeParams.gridDim.y == 1);
  REQUIRE(receivedKernelNodeParams.gridDim.z == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.x == 2);
  REQUIRE(receivedKernelNodeParams.blockDim.y == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.z == 1);
  REQUIRE(*(reinterpret_cast<int*>(receivedKernelNodeParams.kernelParams[0])) ==
          *(reinterpret_cast<int*>(kernelArgs[0])));
  REQUIRE(*(reinterpret_cast<int*>(receivedKernelNodeParams.kernelParams[1])) ==
          *(reinterpret_cast<int*>(kernelArgs[1])));
  REQUIRE(receivedKernelNodeParams.extra == nullptr);
  REQUIRE(receivedKernelNodeParams.sharedMemBytes == 0);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(validateHostArray(hostMem, N, 102) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream))
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (set/get attributes for kernel) APIs from the
 *  - hipGetProcAddress and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_KernelNodeSetGetAttribute") {
  void* hipGraphKernelNodeSetAttribute_ptr = nullptr;
  void* hipGraphKernelNodeGetAttribute_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphKernelNodeSetAttribute", &hipGraphKernelNodeSetAttribute_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphKernelNodeGetAttribute", &hipGraphKernelNodeGetAttribute_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphKernelNodeSetAttribute_ptr)(hipGraphNode_t, hipKernelNodeAttrID,
                                                       const hipKernelNodeAttrValue*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipKernelNodeAttrID,
                                      const hipKernelNodeAttrValue*)>(
          hipGraphKernelNodeSetAttribute_ptr);

  hipError_t (*dyn_hipGraphKernelNodeGetAttribute_ptr)(hipGraphNode_t, hipKernelNodeAttrID,
                                                       hipKernelNodeAttrValue*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipKernelNodeAttrID,
                                      hipKernelNodeAttrValue*)>(hipGraphKernelNodeGetAttribute_ptr);

  hipGraphNode_t kernelNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(simpleKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = nullptr;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));

  hipKernelNodeAttrValue attributeToSet, attributeToGet;
  attributeToSet.cooperative = 1;

  // Validating hipGraphKernelNodeSetAttribute API
  HIP_CHECK(dyn_hipGraphKernelNodeSetAttribute_ptr(kernelNode, hipKernelNodeAttributeCooperative,
                                                   &attributeToSet));

  // Validating hipGraphKernelNodeGetAttribute API
  HIP_CHECK(dyn_hipGraphKernelNodeGetAttribute_ptr(kernelNode, hipKernelNodeAttributeCooperative,
                                                   &attributeToGet));

  REQUIRE(attributeToGet.cooperative == 1);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (host) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_HostNode") {
  void* hipGraphAddHostNode_ptr = nullptr;
  void* hipGraphHostNodeGetParams_ptr = nullptr;
  void* hipGraphHostNodeSetParams_ptr = nullptr;
  void* hipGraphExecHostNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphAddHostNode", &hipGraphAddHostNode_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphHostNodeGetParams", &hipGraphHostNodeGetParams_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphHostNodeSetParams", &hipGraphHostNodeSetParams_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphExecHostNodeSetParams", &hipGraphExecHostNodeSetParams_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddHostNode_ptr)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                                            size_t, const hipHostNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t,
                                      const hipHostNodeParams*)>(hipGraphAddHostNode_ptr);
  hipError_t (*dyn_hipGraphHostNodeGetParams_ptr)(hipGraphNode_t, hipHostNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipHostNodeParams*)>(
          hipGraphHostNodeGetParams_ptr);
  hipError_t (*dyn_hipGraphHostNodeSetParams_ptr)(hipGraphNode_t, const hipHostNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, const hipHostNodeParams*)>(
          hipGraphHostNodeSetParams_ptr);
  hipError_t (*dyn_hipGraphExecHostNodeSetParams_ptr)(hipGraphExec_t, hipGraphNode_t,
                                                      const hipHostNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t, const hipHostNodeParams*)>(
          hipGraphExecHostNodeSetParams_ptr);

  // Validating hipGraphAddHostNode API
  {
    int hostInt = 10;
    hipGraphNode_t hostNode;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Prepare hostNode
    hipHostNodeParams hostNodeParams;
    hostNodeParams.fn = addTen;
    hostNodeParams.userData = &hostInt;

    HIP_CHECK(dyn_hipGraphAddHostNode_ptr(&hostNode, graph, nullptr, 0, &hostNodeParams));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    REQUIRE(hostInt == 20);

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  // Validating hipGraphHostNodeGetParams, hipGraphHostNodeSetParams API's
  {
    int hostInt = 10;
    int hostIntNew = 20;

    hipGraphNode_t hostNode;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Prepare hostNode
    hipHostNodeParams hostNodeParams;
    hostNodeParams.fn = addTen;
    hostNodeParams.userData = &hostInt;

    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostNodeParams));

    hipHostNodeParams receivedHostNodeParams;
    HIP_CHECK(dyn_hipGraphHostNodeGetParams_ptr(hostNode, &receivedHostNodeParams));
    REQUIRE(receivedHostNodeParams.fn == addTen);
    REQUIRE(*(reinterpret_cast<int*>(receivedHostNodeParams.userData)) == 10);

    hipHostNodeParams hostNodeParamsNew;
    hostNodeParamsNew.fn = addTwenty;
    hostNodeParamsNew.userData = &hostIntNew;
    HIP_CHECK(dyn_hipGraphHostNodeSetParams_ptr(hostNode, &hostNodeParamsNew));

    HIP_CHECK(dyn_hipGraphHostNodeGetParams_ptr(hostNode, &receivedHostNodeParams));
    REQUIRE(receivedHostNodeParams.fn == addTwenty);
    REQUIRE(*(reinterpret_cast<int*>(receivedHostNodeParams.userData)) == 20);

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    REQUIRE(hostInt == 10);
    REQUIRE(hostIntNew == 40);

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  // Validating hipGraphExecHostNodeSetParams API
  {
    int hostInt = 10;
    int hostIntNew = 20;

    hipGraphNode_t hostNode;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Prepare hostNode
    hipHostNodeParams hostNodeParams;
    hostNodeParams.fn = addTen;
    hostNodeParams.userData = &hostInt;

    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostNodeParams));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Update hostNode params
    hipHostNodeParams hostNodeParamsNew;
    hostNodeParamsNew.fn = addTwenty;
    hostNodeParamsNew.userData = &hostIntNew;
    HIP_CHECK(dyn_hipGraphExecHostNodeSetParams_ptr(graphExec, hostNode, &hostNodeParamsNew));

    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    REQUIRE(hostInt == 10);
    REQUIRE(hostIntNew == 40);

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipGraphExecUpdate API
 *  - from the hipGetProcAddress API
 *  - and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ExecUpdate") {
  void* hipGraphExecUpdate_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphExecUpdate", &hipGraphExecUpdate_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipGraphExecUpdate_ptr)(hipGraphExec_t, hipGraph_t, hipGraphNode_t*,
                                           hipGraphExecUpdateResult*) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraph_t, hipGraphNode_t*,
                                      hipGraphExecUpdateResult*)>(hipGraphExecUpdate_ptr);

  int hostInt_1 = 10;
  int hostInt_2 = 20;

  // Prepare graph1 with hostNode1
  hipGraph_t graph1 = nullptr;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  hipGraphNode_t hostNode1;
  hipHostNodeParams hostNodeParams1;
  hostNodeParams1.fn = addTen;
  hostNodeParams1.userData = &hostInt_1;
  HIP_CHECK(hipGraphAddHostNode(&hostNode1, graph1, nullptr, 0, &hostNodeParams1));

  // Prepare graphExec with graph1
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));

  // Prepare graph2 with hostNode2
  hipGraph_t graph2 = nullptr;
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  hipGraphNode_t hostNode2;
  hipHostNodeParams hostNodeParams2;
  hostNodeParams2.fn = addTwenty;
  hostNodeParams2.userData = &hostInt_2;
  HIP_CHECK(hipGraphAddHostNode(&hostNode2, graph2, nullptr, 0, &hostNodeParams2));

  // Update graphExec with graph2
  hipGraphNode_t hErrorNode_out = nullptr;
  hipGraphExecUpdateResult updateResult_out;

  // Validating hipGraphExecUpdate API
  HIP_CHECK(dyn_hipGraphExecUpdate_ptr(graphExec, graph2, &hErrorNode_out, &updateResult_out));

  REQUIRE(hErrorNode_out == nullptr);
  REQUIRE(updateResult_out == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  REQUIRE(hostInt_1 == 10);
  REQUIRE(hostInt_2 == 40);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (for mem cpoy node set params) APIs from the
 *  - hipGetProcAddress and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_Memcpy1DSetParams") {
  void* hipGraphMemcpyNodeSetParams1D_ptr = nullptr;
  void* hipGraphExecMemcpyNodeSetParams1D_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphMemcpyNodeSetParams1D", &hipGraphMemcpyNodeSetParams1D_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphExecMemcpyNodeSetParams1D",
                              &hipGraphExecMemcpyNodeSetParams1D_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipGraphMemcpyNodeSetParams1D_ptr)(hipGraphNode_t, void*, const void*, size_t,
                                                      hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, void*, const void*, size_t, hipMemcpyKind)>(
          hipGraphMemcpyNodeSetParams1D_ptr);

  hipError_t (*dyn_hipGraphExecMemcpyNodeSetParams1D_ptr)(hipGraphExec_t, hipGraphNode_t, void*,
                                                          const void*, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t, void*, const void*, size_t,
                                      hipMemcpyKind)>(hipGraphExecMemcpyNodeSetParams1D_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 20);

  // Validating hipGraphMemcpyNodeSetParams1D API
  {
    int* devMem_1 = nullptr;
    HIP_CHECK(hipMalloc(&devMem_1, Nbytes));
    REQUIRE(devMem_1 != nullptr);
    fillDeviceArray(devMem_1, N, 10);

    int* devMem_2 = nullptr;
    HIP_CHECK(hipMalloc(&devMem_2, Nbytes));
    REQUIRE(devMem_2 != nullptr);
    fillDeviceArray(devMem_2, N, 10);

    hipGraphNode_t memcpyNodeH2D;

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0, devMem_1, hostMem, Nbytes,
                                      hipMemcpyHostToDevice));

    HIP_CHECK(dyn_hipGraphMemcpyNodeSetParams1D_ptr(memcpyNodeH2D, devMem_2, hostMem, Nbytes,
                                                    hipMemcpyHostToDevice));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    REQUIRE(validateDeviceArray(devMem_1, N, 10) == true);
    REQUIRE(validateDeviceArray(devMem_2, N, 20) == true);

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipFree(devMem_1));
    HIP_CHECK(hipFree(devMem_2));
  }

  // Validating hipGraphExecMemcpyNodeSetParams1D API
  {
    int* devMem_1 = nullptr;
    HIP_CHECK(hipMalloc(&devMem_1, Nbytes));
    REQUIRE(devMem_1 != nullptr);
    fillDeviceArray(devMem_1, N, 10);

    int* devMem_2 = nullptr;
    HIP_CHECK(hipMalloc(&devMem_2, Nbytes));
    REQUIRE(devMem_2 != nullptr);
    fillDeviceArray(devMem_2, N, 10);

    hipGraphNode_t memcpyNodeH2D;

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0, devMem_1, hostMem, Nbytes,
                                      hipMemcpyHostToDevice));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    HIP_CHECK(dyn_hipGraphExecMemcpyNodeSetParams1D_ptr(graphExec, memcpyNodeH2D, devMem_2, hostMem,
                                                        Nbytes, hipMemcpyHostToDevice));

    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    REQUIRE(validateDeviceArray(devMem_1, N, 10) == true);
    REQUIRE(validateDeviceArray(devMem_2, N, 20) == true);

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipFree(devMem_1));
    HIP_CHECK(hipFree(devMem_2));
  }
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (memory allocation and free) APIs from the
 *  - hipGetProcAddress and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_MemAllocAndFree") {
  void* hipGraphAddMemAllocNode_ptr = nullptr;
  void* hipGraphAddMemFreeNode_ptr = nullptr;
  void* hipGraphMemAllocNodeGetParams_ptr = nullptr;
  void* hipGraphMemFreeNodeGetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphAddMemAllocNode", &hipGraphAddMemAllocNode_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphAddMemFreeNode", &hipGraphAddMemFreeNode_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphMemAllocNodeGetParams", &hipGraphMemAllocNodeGetParams_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphMemFreeNodeGetParams", &hipGraphMemFreeNodeGetParams_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddMemAllocNode_ptr)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                                                size_t, hipMemAllocNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t,
                                      hipMemAllocNodeParams*)>(hipGraphAddMemAllocNode_ptr);

  hipError_t (*dyn_hipGraphAddMemFreeNode_ptr)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                                               size_t, void*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t,
                                      void*)>(hipGraphAddMemFreeNode_ptr);

  hipError_t (*dyn_hipGraphMemAllocNodeGetParams_ptr)(hipGraphNode_t, hipMemAllocNodeParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipMemAllocNodeParams*)>(
          hipGraphMemAllocNodeGetParams_ptr);

  hipError_t (*dyn_hipGraphMemFreeNodeGetParams_ptr)(hipGraphNode_t, void*) =
      reinterpret_cast<hipError_t (*)(hipGraphNode_t, void*)>(hipGraphMemFreeNodeGetParams_ptr);

  int N = 300;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode, kernelNode, memcpyNodeD2H, memFreeNode;

  hipMemAllocNodeParams memAllocNodeParams{};
  memAllocNodeParams.poolProps.allocType = hipMemAllocationTypePinned;
  memAllocNodeParams.poolProps.handleTypes = hipMemHandleTypeNone;
  memAllocNodeParams.poolProps.location.type = hipMemLocationTypeDevice;
  memAllocNodeParams.poolProps.location.id = 0;
  memAllocNodeParams.poolProps.win32SecurityAttributes = nullptr;
  memAllocNodeParams.poolProps.maxSize = 1024;
  hipMemAccessDesc accessDescs;
  accessDescs.location.id = 0;
  accessDescs.location.type = hipMemLocationTypeDevice;
  accessDescs.flags = hipMemAccessFlagsProtReadWrite;
  memAllocNodeParams.accessDescs = &accessDescs;
  memAllocNodeParams.accessDescCount = 1;
  memAllocNodeParams.bytesize = Nbytes;

  // Validating hipGraphAddMemAllocNode API
  HIP_CHECK(dyn_hipGraphAddMemAllocNode_ptr(&memAllocNode, graph, nullptr, 0, &memAllocNodeParams));
  devMem = reinterpret_cast<int*>(memAllocNodeParams.dptr);

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memAllocNode);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(fillArray);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  int value = 20;

  void* kernelArgs[3] = {reinterpret_cast<void*>(&devMem), reinterpret_cast<void*>(&N),
                         reinterpret_cast<void*>(&value)};
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, kernelNodeDependencies.data(),
                                  kernelNodeDependencies.size(), &kernelNodeParams));

  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph, memcpyNodeD2HDependencies.data(),
                                    memcpyNodeD2HDependencies.size(), hostMem, devMem, Nbytes,
                                    hipMemcpyDeviceToHost));

  ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
  memFreeNodeDependencies.push_back(memcpyNodeD2H);

  // Validating hipGraphAddMemFreeNode API
  HIP_CHECK(dyn_hipGraphAddMemFreeNode_ptr(&memFreeNode, graph, memFreeNodeDependencies.data(),
                                           memFreeNodeDependencies.size(),
                                           reinterpret_cast<void*>(devMem)));

  // Validating hipGraphMemAllocNodeGetParams API
  hipMemAllocNodeParams recvdParams{};
  HIP_CHECK(dyn_hipGraphMemAllocNodeGetParams_ptr(memAllocNode, &recvdParams));

  REQUIRE(recvdParams.poolProps.allocType == hipMemAllocationTypePinned);
  REQUIRE(recvdParams.poolProps.handleTypes == hipMemHandleTypeNone);
  REQUIRE(recvdParams.poolProps.location.type == hipMemLocationTypeDevice);
  REQUIRE(recvdParams.poolProps.location.id == 0);
  REQUIRE(recvdParams.poolProps.win32SecurityAttributes == nullptr);
  REQUIRE(recvdParams.poolProps.maxSize == 1024);
  REQUIRE(recvdParams.accessDescs->location.id == 0);
  REQUIRE(recvdParams.accessDescs->location.type == hipMemLocationTypeDevice);
  REQUIRE(recvdParams.accessDescs->flags == hipMemAccessFlagsProtReadWrite);
  REQUIRE(recvdParams.accessDescCount == 1);
  REQUIRE(recvdParams.bytesize == Nbytes);

  // Validating hipGraphMemFreeNodeGetParams API
  size_t dev_ptr = 0;
  HIP_CHECK(dyn_hipGraphMemFreeNodeGetParams_ptr(memFreeNode, reinterpret_cast<void*>(&dev_ptr)));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  REQUIRE(validateHostArray(hostMem, N, 20) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (for memset node set params) APIs from the
 *  - hipGetProcAddress and then validates the basic functionality of that
 *  - particular APIusing the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Graph_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ExecMemsetMemcpySetParams") {
  CHECK_IMAGE_SUPPORT

  void* hipGraphExecMemsetNodeSetParams_ptr = nullptr;
  void* hipGraphExecMemcpyNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGraphExecMemsetNodeSetParams",
                              &hipGraphExecMemsetNodeSetParams_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphExecMemcpyNodeSetParams",
                              &hipGraphExecMemcpyNodeSetParams_ptr, currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphExecMemsetNodeSetParams_ptr)(hipGraphExec_t, hipGraphNode_t,
                                                        const hipMemsetParams*) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t, const hipMemsetParams*)>(
          hipGraphExecMemsetNodeSetParams_ptr);

  hipError_t (*dyn_hipGraphExecMemcpyNodeSetParams_ptr)(hipGraphExec_t, hipGraphNode_t,
                                                        hipMemcpy3DParms*) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t, hipMemcpy3DParms*)>(
          hipGraphExecMemcpyNodeSetParams_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char* devMemSrc1 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc1), &pitch, width, height));
  REQUIRE(devMemSrc1 != nullptr);

  char* hostMemDst1 = reinterpret_cast<char*>(malloc(N * sizeof(char)));
  REQUIRE(hostMemDst1 != nullptr);
  fillCharHostArray(hostMemDst1, N, 100);

  char* devMemSrc2 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc2), &pitch, width, height));
  REQUIRE(devMemSrc2 != nullptr);

  char* hostMemDst2 = reinterpret_cast<char*>(malloc(N * sizeof(char)));
  REQUIRE(hostMemDst2 != nullptr);
  fillCharHostArray(hostMemDst2, N, 100);

  hipGraphNode_t memsetNode, memcpyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memset node
  hipMemsetParams initialMemsetParams{};
  initialMemsetParams.dst = reinterpret_cast<void*>(devMemSrc1);
  initialMemsetParams.value = value;
  initialMemsetParams.pitch = pitch;
  initialMemsetParams.elementSize = sizeof(char);
  initialMemsetParams.width = width;
  initialMemsetParams.height = height;

  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &initialMemsetParams));

  // Prepare memcpyNode
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  hipMemcpy3DParms initialParms{};
  initialParms.srcPos = make_hipPos(0, 0, 0);
  initialParms.dstPos = make_hipPos(0, 0, 0);
  initialParms.srcPtr = make_hipPitchedPtr(devMemSrc1, pitch, width, height);
  initialParms.dstPtr = make_hipPitchedPtr(hostMemDst1, width, width, height);
  initialParms.extent = make_hipExtent(width, height, 1);
  initialParms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, memcpyNodeDependencies.data(),
                                  memcpyNodeDependencies.size(), &initialParms));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipMemsetParams newMemsetParams{};
  newMemsetParams.dst = reinterpret_cast<void*>(devMemSrc2);
  newMemsetParams.value = value;
  newMemsetParams.pitch = pitch;
  newMemsetParams.elementSize = sizeof(char);
  newMemsetParams.width = width;
  newMemsetParams.height = height;

  // Validating hipGraphExecMemsetNodeSetParams API
  HIP_CHECK(dyn_hipGraphExecMemsetNodeSetParams_ptr(graphExec, memsetNode, &newMemsetParams));

  hipMemcpy3DParms newMemcpyParms{};
  newMemcpyParms.srcPos = make_hipPos(0, 0, 0);
  newMemcpyParms.dstPos = make_hipPos(0, 0, 0);
  newMemcpyParms.srcPtr = make_hipPitchedPtr(devMemSrc2, pitch, width, height);
  newMemcpyParms.dstPtr = make_hipPitchedPtr(hostMemDst2, width, width, height);
  newMemcpyParms.extent = make_hipExtent(width, height, 1);
  newMemcpyParms.kind = hipMemcpyDeviceToHost;

  // Validating hipGraphExecMemcpyNodeSetParams API
  HIP_CHECK(dyn_hipGraphExecMemcpyNodeSetParams_ptr(graphExec, memcpyNode, &newMemcpyParms));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  REQUIRE(validateArrayT<char>(hostMemDst1, N, 100) == true);
  REQUIRE(validateArrayT<char>(hostMemDst2, N, 120) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMemSrc1));
  HIP_CHECK(hipFree(devMemSrc2));
  free(hostMemDst1);
  free(hostMemDst2);
}
