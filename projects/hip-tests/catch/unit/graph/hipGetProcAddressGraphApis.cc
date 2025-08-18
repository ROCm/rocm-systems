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
#include <fstream>
#include <filesystem>
#include "../device/hipGetProcAddressHelpers.hh"

/**
 * Local to destroy user object, used in user object scenarios
 */
void destroyIntObject(void* obj) {
  int* ptr = reinterpret_cast<int*>(obj);
  delete ptr;
}

const int N = 1024;
__device__ int symbolData1[N];
__device__ int symbolData2[N];
__device__ int symbolData3[N];
const int Nbytes = N * sizeof(int);

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph (stream capture) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
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

  HIP_CHECK(hipGetProcAddress(
            "hipStreamBeginCapture",
            &hipStreamBeginCapture_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipStreamIsCapturing",
            &hipStreamIsCapturing_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipStreamEndCapture",
            &hipStreamEndCapture_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipStreamAddCallback",
            &hipStreamAddCallback_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphInstantiate",
            &hipGraphInstantiate_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphLaunch",
            &hipGraphLaunch_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecDestroy",
            &hipGraphExecDestroy_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipStreamBeginCapture_ptr)(
    hipStream_t, hipStreamCaptureMode) =
    reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureMode)>
    (hipStreamBeginCapture_ptr);

  hipError_t (*dyn_hipStreamIsCapturing_ptr)(hipStream_t,
    hipStreamCaptureStatus *) =
    reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureStatus *)>
    (hipStreamIsCapturing_ptr);

  hipError_t (*dyn_hipStreamEndCapture_ptr)(hipStream_t, hipGraph_t *) =
    reinterpret_cast<hipError_t (*)(hipStream_t, hipGraph_t *)>
    (hipStreamEndCapture_ptr);

  hipError_t (*dyn_hipStreamAddCallback_ptr)(hipStream_t, hipStreamCallback_t,
                                             void *, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCallback_t,
                                    void *, unsigned int)>
                                   (hipStreamAddCallback_ptr);

  hipError_t (*dyn_hipGraphInstantiate_ptr)(hipGraphExec_t *, hipGraph_t,
                                            hipGraphNode_t *, char *, size_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t *, hipGraph_t,
                                    hipGraphNode_t *, char *, size_t)>
                                   (hipGraphInstantiate_ptr);

  hipError_t (*dyn_hipGraphLaunch_ptr)(hipGraphExec_t, hipStream_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipStream_t)>
    (hipGraphLaunch_ptr);

  hipError_t (*dyn_hipGraphExecDestroy_ptr)(hipGraphExec_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t)>
    (hipGraphExecDestroy_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Validating hipStreamBeginCapture API
  HIP_CHECK(dyn_hipStreamBeginCapture_ptr(stream,
                                          hipStreamCaptureModeGlobal));

  // Validating hipStreamIsCapturing API
  hipStreamCaptureStatus pCaptureStatus = hipStreamCaptureStatusNone;
  HIP_CHECK(dyn_hipStreamIsCapturing_ptr(stream, &pCaptureStatus));
  REQUIRE(pCaptureStatus == hipStreamCaptureStatusActive);

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes,
                           hipMemcpyHostToDevice, stream));
  addOneKernel<<< 1, 1, 0, stream >>>(devMem , N);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  // Validating hipStreamEndCapture API
  HIP_CHECK(dyn_hipStreamEndCapture_ptr(stream, &graph));

  // Validating hipStreamAddCallback API
  int data = 100;
  HIP_CHECK(dyn_hipStreamAddCallback_ptr(stream, callBackFunction,
            reinterpret_cast<void *>(&data), 0));

  // Validating hipGraphInstantiate API
  hipGraphExec_t graphExec;
  HIP_CHECK(dyn_hipGraphInstantiate_ptr(&graphExec, graph,
                                        nullptr, nullptr, 0));

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_AddMemcpy1DKernelNodes") {
  void* hipGraphAddMemcpyNode1D_ptr = nullptr;
  void* hipGraphAddKernelNode_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddMemcpyNode1D",
            &hipGraphAddMemcpyNode1D_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddKernelNode",
            &hipGraphAddKernelNode_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddMemcpyNode1D_ptr)(hipGraphNode_t *, hipGraph_t,
             const hipGraphNode_t *, size_t, void *, const void *, size_t,
             hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, void *,
    const void *, size_t, hipMemcpyKind)>
    (hipGraphAddMemcpyNode1D_ptr);

  hipError_t (*dyn_hipGraphAddKernelNode_ptr)(hipGraphNode_t *, hipGraph_t,
              const hipGraphNode_t *, size_t, const hipKernelNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const hipKernelNodeParams *)>
    (hipGraphAddKernelNode_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
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
  HIP_CHECK(dyn_hipGraphAddMemcpyNode1D_ptr(&memcpyNodeH2D, graph, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  // Validating hipGraphAddKernelNode API
  // Prepare kernelNode with memcpyNodeH2D as a dependency
  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(dyn_hipGraphAddKernelNode_ptr(&kernelNode, graph,
            kernelNodeDependencies.data(), kernelNodeDependencies.size(),
            &kernelNodeParams));

  // Prepare memcpyNodeD2H with kernelNode as a dependency
  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);
  HIP_CHECK(dyn_hipGraphAddMemcpyNode1D_ptr(&memcpyNodeD2H, graph,
            memcpyNodeD2HDependencies.data(), memcpyNodeD2HDependencies.size(),
            hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
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

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddMemsetNode",
            &hipGraphAddMemsetNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddMemcpyNode",
            &hipGraphAddMemcpyNode_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddMemsetNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const hipMemsetParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const hipMemsetParams *)>
    (hipGraphAddMemsetNode_ptr);

  hipError_t (*dyn_hipGraphAddMemcpyNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const hipMemcpy3DParms *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const hipMemcpy3DParms *)>
    (hipGraphAddMemcpyNode_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char *devMemSrc = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc),
                             &pitch, width, height));
  REQUIRE(devMemSrc != nullptr);

  char* hostMemDst = reinterpret_cast<char *>(malloc( N * sizeof(char)));
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

  HIP_CHECK(dyn_hipGraphAddMemsetNode_ptr(&memsetNode, graph,
                                          nullptr, 0, &pMemsetParams));

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
  HIP_CHECK(dyn_hipGraphAddMemcpyNode_ptr(&memcpyNode, graph,
                                          memcpyNodeDependencies.data(),
                                          memcpyNodeDependencies.size(),
                                          &myparms));

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
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

  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemsetNodeSetParams",
            &hipGraphMemsetNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemsetNodeGetParams",
            &hipGraphMemsetNodeGetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemcpyNodeSetParams",
            &hipGraphMemcpyNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemcpyNodeGetParams",
            &hipGraphMemcpyNodeGetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphMemsetNodeSetParams_ptr)(
    hipGraphNode_t, const hipMemsetParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, const hipMemsetParams *)>
    (hipGraphMemsetNodeSetParams_ptr);

  hipError_t (*dyn_hipGraphMemsetNodeGetParams_ptr)(
    hipGraphNode_t, hipMemsetParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipMemsetParams *)>
    (hipGraphMemsetNodeGetParams_ptr);

  hipError_t (*dyn_hipGraphMemcpyNodeSetParams_ptr)(
    hipGraphNode_t, const hipMemcpy3DParms *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, const hipMemcpy3DParms *)>
    (hipGraphMemcpyNodeSetParams_ptr);

  hipError_t (*dyn_hipGraphMemcpyNodeGetParams_ptr)(
    hipGraphNode_t, hipMemcpy3DParms *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipMemcpy3DParms *)>
    (hipGraphMemcpyNodeGetParams_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char *devMemSrc1 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc1),
                             &pitch, width, height));
  REQUIRE(devMemSrc1 != nullptr);

  char* hostMemDst1 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst1 != nullptr);
  fillCharHostArray(hostMemDst1, N, 100);

  char *devMemSrc2 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc2),
                             &pitch, width, height));
  REQUIRE(devMemSrc2 != nullptr);

  char* hostMemDst2 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
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

  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph,
                                   nullptr, 0, &initialMemsetParams));

  hipMemsetParams receivedMemsetValues{};
  HIP_CHECK(dyn_hipGraphMemsetNodeGetParams_ptr(memsetNode,
                                                &receivedMemsetValues));

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
  HIP_CHECK(dyn_hipGraphMemsetNodeSetParams_ptr(memsetNode,
                                                &correctedMemsetParams));

  // Validating hipGraphMemsetNodeGetParams API
  HIP_CHECK(dyn_hipGraphMemsetNodeGetParams_ptr(memsetNode,
                                                &receivedMemsetValues));

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
  initialParms.srcPtr = make_hipPitchedPtr(devMemSrc1, pitch,
                                           width, height);
  initialParms.dstPtr = make_hipPitchedPtr(hostMemDst1, width,
                                           width, height);
  initialParms.extent = make_hipExtent(width, height, 1);
  initialParms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph,
            memcpyNodeDependencies.data(),
            memcpyNodeDependencies.size(), &initialParms));

  hipMemcpy3DParms receivedMemcpyValues{};
  HIP_CHECK(dyn_hipGraphMemcpyNodeGetParams_ptr(memcpyNode,
                                                &receivedMemcpyValues));

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
  correctedParms.dstPtr = make_hipPitchedPtr(hostMemDst2, width,
                                             width, height);
  correctedParms.extent = make_hipExtent(width, height, 1);
  correctedParms.kind = hipMemcpyDeviceToHost;

  // Validating hipGraphMemcpyNodeSetParams API
  HIP_CHECK(dyn_hipGraphMemcpyNodeSetParams_ptr(memcpyNode, &correctedParms));

  // Validating hipGraphMemcpyNodeGetParams API
  HIP_CHECK(dyn_hipGraphMemcpyNodeGetParams_ptr(memcpyNode,
                                                &receivedMemcpyValues));

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
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_KernelNodeSetGetParams") {
  void* hipGraphKernelNodeGetParams_ptr = nullptr;
  void* hipGraphKernelNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphKernelNodeGetParams",
            &hipGraphKernelNodeGetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphKernelNodeSetParams",
            &hipGraphKernelNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphKernelNodeGetParams_ptr)(
    hipGraphNode_t, hipKernelNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipKernelNodeParams *)>
    (hipGraphKernelNodeGetParams_ptr);

  hipError_t (*dyn_hipGraphKernelNodeSetParams_ptr)(
    hipGraphNode_t, const hipKernelNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t,
    const hipKernelNodeParams *)>(hipGraphKernelNodeSetParams_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
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
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  // Prepare kernelNode with memcpyNodeH2D as a dependency
  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
            kernelNodeDependencies.data(),
            kernelNodeDependencies.size(), &kernelNodeParams));

  // Prepare memcpyNodeD2H with kernelNode as a dependency
  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph,
            memcpyNodeD2HDependencies.data(), memcpyNodeD2HDependencies.size(),
            hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

  // Get and set Kernel node params
  hipKernelNodeParams receivedKernelNodeParams;

  // Validating hipGraphKernelNodeGetParams API
  HIP_CHECK(dyn_hipGraphKernelNodeGetParams_ptr(kernelNode,
                                                &receivedKernelNodeParams));

  REQUIRE(receivedKernelNodeParams.func == addOneKernel);
  REQUIRE(receivedKernelNodeParams.gridDim.x == 1);
  REQUIRE(receivedKernelNodeParams.gridDim.y == 1);
  REQUIRE(receivedKernelNodeParams.gridDim.z == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.x == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.y == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.z == 1);
  REQUIRE(*(reinterpret_cast<int *>(receivedKernelNodeParams.kernelParams[0]))
          == *(reinterpret_cast<int *>(kernelArgs[0])));
  REQUIRE(*(reinterpret_cast<int *>(receivedKernelNodeParams.kernelParams[1]))
          == *(reinterpret_cast<int *>(kernelArgs[1])));
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
  HIP_CHECK(dyn_hipGraphKernelNodeSetParams_ptr(kernelNode,
                                                &correctedKernelNodeParams));

  HIP_CHECK(dyn_hipGraphKernelNodeGetParams_ptr(kernelNode,
                                                &receivedKernelNodeParams));

  REQUIRE(receivedKernelNodeParams.func == addTwoKernel);
  REQUIRE(receivedKernelNodeParams.gridDim.x == 2);
  REQUIRE(receivedKernelNodeParams.gridDim.y == 1);
  REQUIRE(receivedKernelNodeParams.gridDim.z == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.x == 2);
  REQUIRE(receivedKernelNodeParams.blockDim.y == 1);
  REQUIRE(receivedKernelNodeParams.blockDim.z == 1);
  REQUIRE(*(reinterpret_cast<int *>(receivedKernelNodeParams.kernelParams[0]))
          == *(reinterpret_cast<int *>(kernelArgs[0])));
  REQUIRE(*(reinterpret_cast<int *>(receivedKernelNodeParams.kernelParams[1]))
          == *(reinterpret_cast<int *>(kernelArgs[1])));
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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_KernelNodeSetGetAttribute") {
  void* hipGraphKernelNodeSetAttribute_ptr = nullptr;
  void* hipGraphKernelNodeGetAttribute_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphKernelNodeSetAttribute",
            &hipGraphKernelNodeSetAttribute_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphKernelNodeGetAttribute",
            &hipGraphKernelNodeGetAttribute_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphKernelNodeSetAttribute_ptr)(
    hipGraphNode_t, hipKernelNodeAttrID, const hipKernelNodeAttrValue *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipKernelNodeAttrID,
    const hipKernelNodeAttrValue *)>(hipGraphKernelNodeSetAttribute_ptr);

  hipError_t (*dyn_hipGraphKernelNodeGetAttribute_ptr)(hipGraphNode_t,
    hipKernelNodeAttrID, hipKernelNodeAttrValue *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipKernelNodeAttrID,
    hipKernelNodeAttrValue *)>(hipGraphKernelNodeGetAttribute_ptr);

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

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
            nullptr, 0, &kernelNodeParams));

  hipKernelNodeAttrValue attributeToSet, attributeToGet;
  attributeToSet.cooperative = 1;

  // Validating hipGraphKernelNodeSetAttribute API
  HIP_CHECK(dyn_hipGraphKernelNodeSetAttribute_ptr(kernelNode,
            hipKernelNodeAttributeCooperative, &attributeToSet));

  // Validating hipGraphKernelNodeGetAttribute API
  HIP_CHECK(dyn_hipGraphKernelNodeGetAttribute_ptr(kernelNode,
            hipKernelNodeAttributeCooperative, &attributeToGet));

  REQUIRE(attributeToGet.cooperative == 1);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
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

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddHostNode",
            &hipGraphAddHostNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphHostNodeGetParams",
            &hipGraphHostNodeGetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphHostNodeSetParams",
            &hipGraphHostNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecHostNodeSetParams",
            &hipGraphExecHostNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddHostNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const hipHostNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const hipHostNodeParams *)>
    (hipGraphAddHostNode_ptr);
  hipError_t (*dyn_hipGraphHostNodeGetParams_ptr)(hipGraphNode_t,
    hipHostNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipHostNodeParams *)>
    (hipGraphHostNodeGetParams_ptr);
  hipError_t (*dyn_hipGraphHostNodeSetParams_ptr)(hipGraphNode_t,
    const hipHostNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, const hipHostNodeParams *)>
    (hipGraphHostNodeSetParams_ptr);
  hipError_t (*dyn_hipGraphExecHostNodeSetParams_ptr)(hipGraphExec_t,
    hipGraphNode_t, const hipHostNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
    const hipHostNodeParams *)>
    (hipGraphExecHostNodeSetParams_ptr);

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

    HIP_CHECK(dyn_hipGraphAddHostNode_ptr(&hostNode, graph,
              nullptr, 0, &hostNodeParams));

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

    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph,
              nullptr, 0, &hostNodeParams));

    hipHostNodeParams receivedHostNodeParams;
    HIP_CHECK(dyn_hipGraphHostNodeGetParams_ptr(hostNode,
                                                &receivedHostNodeParams));
    REQUIRE(receivedHostNodeParams.fn == addTen);
    REQUIRE(*(reinterpret_cast<int *>(receivedHostNodeParams.userData)) == 10);

    hipHostNodeParams hostNodeParamsNew;
    hostNodeParamsNew.fn = addTwenty;
    hostNodeParamsNew.userData = &hostIntNew;
    HIP_CHECK(dyn_hipGraphHostNodeSetParams_ptr(hostNode, &hostNodeParamsNew));

    HIP_CHECK(dyn_hipGraphHostNodeGetParams_ptr(hostNode,
                                                &receivedHostNodeParams));
    REQUIRE(receivedHostNodeParams.fn == addTwenty);
    REQUIRE(*(reinterpret_cast<int *>(receivedHostNodeParams.userData)) == 20);

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

    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph,
              nullptr, 0, &hostNodeParams));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Update hostNode params
    hipHostNodeParams hostNodeParamsNew;
    hostNodeParamsNew.fn = addTwenty;
    hostNodeParamsNew.userData = &hostIntNew;
    HIP_CHECK(dyn_hipGraphExecHostNodeSetParams_ptr(graphExec, hostNode,
                                                    &hostNodeParamsNew));

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ExecUpdate") {
  void* hipGraphExecUpdate_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecUpdate",
            &hipGraphExecUpdate_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphExecUpdate_ptr)(hipGraphExec_t, hipGraph_t,
    hipGraphNode_t *, hipGraphExecUpdateResult *) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraph_t,
    hipGraphNode_t *, hipGraphExecUpdateResult *)>
    (hipGraphExecUpdate_ptr);

  int hostInt_1 = 10;
  int hostInt_2 = 20;

  // Prepare graph1 with hostNode1
  hipGraph_t graph1 = nullptr;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  hipGraphNode_t hostNode1;
  hipHostNodeParams hostNodeParams1;
  hostNodeParams1.fn = addTen;
  hostNodeParams1.userData = &hostInt_1;
  HIP_CHECK(hipGraphAddHostNode(&hostNode1, graph1,
            nullptr, 0, &hostNodeParams1));

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
  HIP_CHECK(hipGraphAddHostNode(&hostNode2, graph2,
            nullptr, 0, &hostNodeParams2));

  // Update graphExec with graph2
  hipGraphNode_t hErrorNode_out = nullptr;
  hipGraphExecUpdateResult updateResult_out;

  // Validating hipGraphExecUpdate API
  HIP_CHECK(dyn_hipGraphExecUpdate_ptr(graphExec, graph2,
            &hErrorNode_out, &updateResult_out));

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_Memcpy1DSetParams") {
  void* hipGraphMemcpyNodeSetParams1D_ptr = nullptr;
  void* hipGraphExecMemcpyNodeSetParams1D_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemcpyNodeSetParams1D",
            &hipGraphMemcpyNodeSetParams1D_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecMemcpyNodeSetParams1D",
            &hipGraphExecMemcpyNodeSetParams1D_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphMemcpyNodeSetParams1D_ptr)(
    hipGraphNode_t, void *, const void *, size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, void *,
    const void *, size_t, hipMemcpyKind)>
    (hipGraphMemcpyNodeSetParams1D_ptr);

  hipError_t (*dyn_hipGraphExecMemcpyNodeSetParams1D_ptr)(
    hipGraphExec_t, hipGraphNode_t, void *, const void *,
    size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
    void *, const void *, size_t, hipMemcpyKind)>
    (hipGraphExecMemcpyNodeSetParams1D_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
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

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0,
              devMem_1, hostMem, Nbytes, hipMemcpyHostToDevice));

    HIP_CHECK(dyn_hipGraphMemcpyNodeSetParams1D_ptr(memcpyNodeH2D, devMem_2,
              hostMem, Nbytes, hipMemcpyHostToDevice));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    #ifdef _WIN32
    HIP_CHECK(hipStreamSynchronize(0));
    #endif

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

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0,
              devMem_1, hostMem, Nbytes, hipMemcpyHostToDevice));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    HIP_CHECK(dyn_hipGraphExecMemcpyNodeSetParams1D_ptr(graphExec,
              memcpyNodeH2D, devMem_2, hostMem,
              Nbytes, hipMemcpyHostToDevice));

    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    #ifdef _WIN32
    HIP_CHECK(hipStreamSynchronize(0));
    #endif

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
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

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddMemAllocNode",
            &hipGraphAddMemAllocNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddMemFreeNode",
            &hipGraphAddMemFreeNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemAllocNodeGetParams",
            &hipGraphMemAllocNodeGetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemFreeNodeGetParams",
            &hipGraphMemFreeNodeGetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddMemAllocNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, hipMemAllocNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, hipMemAllocNodeParams *)>
    (hipGraphAddMemAllocNode_ptr);

  hipError_t (*dyn_hipGraphAddMemFreeNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, void *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, void *)>
    (hipGraphAddMemFreeNode_ptr);

  hipError_t (*dyn_hipGraphMemAllocNodeGetParams_ptr)(hipGraphNode_t,
    hipMemAllocNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipMemAllocNodeParams *)>
    (hipGraphMemAllocNodeGetParams_ptr);

  hipError_t (*dyn_hipGraphMemFreeNodeGetParams_ptr)(hipGraphNode_t, void *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, void *)>
    (hipGraphMemFreeNodeGetParams_ptr);

  int N = 300;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int *devMem = nullptr;

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
  HIP_CHECK(dyn_hipGraphAddMemAllocNode_ptr(&memAllocNode, graph,
                                    nullptr, 0, &memAllocNodeParams));
  devMem = reinterpret_cast<int*>(memAllocNodeParams.dptr);

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memAllocNode);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(fillArray);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  int value = 20;

  void* kernelArgs[3] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N),
                          reinterpret_cast<void*>(&value) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
            kernelNodeDependencies.data(), kernelNodeDependencies.size(),
            &kernelNodeParams));

  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph,
            memcpyNodeD2HDependencies.data(), memcpyNodeD2HDependencies.size(),
            hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

  ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
  memFreeNodeDependencies.push_back(memcpyNodeD2H);

  // Validating hipGraphAddMemFreeNode API
  HIP_CHECK(dyn_hipGraphAddMemFreeNode_ptr(&memFreeNode, graph,
            memFreeNodeDependencies.data(), memFreeNodeDependencies.size(),
            reinterpret_cast<void*>(devMem)));

  // Validating hipGraphMemAllocNodeGetParams API
  hipMemAllocNodeParams recvdParams{};
  HIP_CHECK(dyn_hipGraphMemAllocNodeGetParams_ptr(memAllocNode,
                                                  &recvdParams));

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
  HIP_CHECK(dyn_hipGraphMemFreeNodeGetParams_ptr(memFreeNode,
            reinterpret_cast<void *>(&dev_ptr)));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

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
 *  - unit/graph/hipGetProcAddressGraphApis.cc
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

  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecMemsetNodeSetParams",
            &hipGraphExecMemsetNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecMemcpyNodeSetParams",
            &hipGraphExecMemcpyNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphExecMemsetNodeSetParams_ptr)(hipGraphExec_t,
    hipGraphNode_t, const hipMemsetParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
    const hipMemsetParams *)>(hipGraphExecMemsetNodeSetParams_ptr);

  hipError_t (*dyn_hipGraphExecMemcpyNodeSetParams_ptr)(hipGraphExec_t,
    hipGraphNode_t, hipMemcpy3DParms *) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
    hipMemcpy3DParms *)>(hipGraphExecMemcpyNodeSetParams_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char *devMemSrc1 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc1),
                             &pitch, width, height));
  REQUIRE(devMemSrc1 != nullptr);

  char* hostMemDst1 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst1 != nullptr);
  fillCharHostArray(hostMemDst1, N, 100);

  char *devMemSrc2 = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc2),
                             &pitch, width, height));
  REQUIRE(devMemSrc2 != nullptr);

  char* hostMemDst2 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
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

  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph,
            nullptr, 0, &initialMemsetParams));

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
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph,
            memcpyNodeDependencies.data(),
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
  HIP_CHECK(dyn_hipGraphExecMemsetNodeSetParams_ptr(graphExec, memsetNode,
                                                    &newMemsetParams));

  hipMemcpy3DParms newMemcpyParms{};
  newMemcpyParms.srcPos = make_hipPos(0, 0, 0);
  newMemcpyParms.dstPos = make_hipPos(0, 0, 0);
  newMemcpyParms.srcPtr = make_hipPitchedPtr(devMemSrc2, pitch, width, height);
  newMemcpyParms.dstPtr = make_hipPitchedPtr(hostMemDst2, width,
                                             width, height);
  newMemcpyParms.extent = make_hipExtent(width, height, 1);
  newMemcpyParms.kind = hipMemcpyDeviceToHost;

  // Validating hipGraphExecMemcpyNodeSetParams API
  HIP_CHECK(dyn_hipGraphExecMemcpyNodeSetParams_ptr(graphExec, memcpyNode,
                                                    &newMemcpyParms));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

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
 *  - This test will get the function pointer of hipGraphAddNode API,
 *  - hipGraphInstantiateWithParams API from the hipGetProcAddress API
 *  - and then validates the basic functionality of those APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_AddNodeAndInstantiateWithParams") {
  void* hipGraphAddNode_ptr = nullptr;
  void* hipGraphInstantiateWithParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddNode",
            &hipGraphAddNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphInstantiateWithParams",
            &hipGraphInstantiateWithParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddNode_ptr)(
    hipGraphNode_t *, hipGraph_t, const hipGraphNode_t *,
    size_t, hipGraphNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
                                    const hipGraphNode_t *,
                                    size_t, hipGraphNodeParams *)>
                                   (hipGraphAddNode_ptr);
  hipError_t (*dyn_hipGraphInstantiateWithParams_ptr)(
    hipGraphExec_t*, hipGraph_t, hipGraphInstantiateParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t*, hipGraph_t,
                                    hipGraphInstantiateParams *)>
                                   (hipGraphInstantiateWithParams_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 10);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t kernelNode;
  hipGraphNodeParams kernelNodeParams = {};
  kernelNodeParams.type = hipGraphNodeTypeKernel;
  kernelNodeParams.kernel.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.kernel.gridDim = dim3(1, 1, 1);
  kernelNodeParams.kernel.blockDim = dim3(1, 1, 1);
  kernelNodeParams.kernel.sharedMemBytes = 0;
  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernel.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.kernel.extra = nullptr;

  // Validating hipGraphAddNode API
  HIP_CHECK(dyn_hipGraphAddNode_ptr(&kernelNode, graph, nullptr, 0,
                                    &kernelNodeParams));

  // Validating hipGraphInstantiateWithParams API
  hipGraphExec_t graphExec;
  hipGraphInstantiateParams iniParams;
  iniParams.flags = 0;
  HIP_CHECK(dyn_hipGraphInstantiateWithParams_ptr(&graphExec, graph,
                                                  &iniParams));
  REQUIRE(iniParams.result_out == hipGraphInstantiateSuccess);
  REQUIRE(iniParams.errNode_out == nullptr);

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem, N, 11) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipStreamBeginCaptureToGraph API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_hipStreamBeginCaptureToGraph") {
  void* hipStreamBeginCaptureToGraph_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipStreamBeginCaptureToGraph",
            &hipStreamBeginCaptureToGraph_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipStreamBeginCaptureToGraph_ptr)(
              hipStream_t, hipGraph_t,
              const hipGraphNode_t*,
              const hipGraphEdgeData*,
              size_t, hipStreamCaptureMode) =
    reinterpret_cast<hipError_t (*)(hipStream_t, hipGraph_t,
                                    const hipGraphNode_t*,
                                    const hipGraphEdgeData*,
                                    size_t, hipStreamCaptureMode)>
                                   (hipStreamBeginCaptureToGraph_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 10);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t kernelNode;
  hipGraphNodeParams kernelNodeParams = {};
  kernelNodeParams.type = hipGraphNodeTypeKernel;
  kernelNodeParams.kernel.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.kernel.gridDim = dim3(1, 1, 1);
  kernelNodeParams.kernel.blockDim = dim3(1, 1, 1);
  kernelNodeParams.kernel.sharedMemBytes = 0;
  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernel.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.kernel.extra = nullptr;
  HIP_CHECK(hipGraphAddNode(&kernelNode, graph, nullptr, 0,
                            &kernelNodeParams));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  ::std::vector<hipGraphNode_t> dependNodes;
  dependNodes.push_back(kernelNode);

  // Validating hipStreamBeginCaptureToGraph API
  HIP_CHECK(dyn_hipStreamBeginCaptureToGraph_ptr(stream, graph,
            dependNodes.data(), nullptr, dependNodes.size(),
            hipStreamCaptureModeGlobal));

  addTwoKernel<<< 1, 1, 0, stream >>>(devMem , N);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph,
                                nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(validateDeviceArray(devMem, N, 13) == true);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph APIs related to Dependencies from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_Dependencies") {
  void* hipGraphAddDependencies_ptr = nullptr;
  void* hipGraphNodeGetDependencies_ptr = nullptr;
  void* hipGraphNodeGetDependentNodes_ptr = nullptr;
  void* hipGraphRemoveDependencies_ptr = nullptr;
  void* hipGraphGetEdges_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddDependencies",
            &hipGraphAddDependencies_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphNodeGetDependencies",
            &hipGraphNodeGetDependencies_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphNodeGetDependentNodes",
            &hipGraphNodeGetDependentNodes_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphRemoveDependencies",
            &hipGraphRemoveDependencies_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphGetEdges",
            &hipGraphGetEdges_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddDependencies_ptr)(
              hipGraph_t, const hipGraphNode_t *,
              const hipGraphNode_t *, size_t) =
    reinterpret_cast<hipError_t (*)(hipGraph_t, const hipGraphNode_t *,
                                    const hipGraphNode_t *, size_t)>
                                   (hipGraphAddDependencies_ptr);

  hipError_t (*dyn_hipGraphNodeGetDependencies_ptr)(
              hipGraphNode_t, hipGraphNode_t *, size_t *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipGraphNode_t *,
                                    size_t *)>
                                   (hipGraphNodeGetDependencies_ptr);

  hipError_t (*dyn_hipGraphNodeGetDependentNodes_ptr)(
              hipGraphNode_t, hipGraphNode_t *, size_t *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipGraphNode_t *,
                                    size_t *)>
                                   (hipGraphNodeGetDependentNodes_ptr);

  hipError_t (*dyn_hipGraphRemoveDependencies_ptr)(
              hipGraph_t, const hipGraphNode_t *,
              const hipGraphNode_t *, size_t) =
    reinterpret_cast<hipError_t (*)(hipGraph_t, const hipGraphNode_t *,
                                    const hipGraphNode_t *, size_t)>
                                   (hipGraphRemoveDependencies_ptr);

  hipError_t (*dyn_hipGraphGetEdges_ptr)(
              hipGraph_t, hipGraphNode_t *, hipGraphNode_t *, size_t *) =
    reinterpret_cast<hipError_t (*)(hipGraph_t, hipGraphNode_t *,
                                    hipGraphNode_t *, size_t *)>
                                   (hipGraphGetEdges_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 100);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t memcpyNodeH2D, kernelNode, memcpyNodeD2H;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
            nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, graph,
            nullptr, 0, hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

  // Validating hipGraphAddDependencies API
  HIP_CHECK(dyn_hipGraphAddDependencies_ptr(graph, &memcpyNodeH2D,
                                            &kernelNode, 1));
  HIP_CHECK(dyn_hipGraphAddDependencies_ptr(graph, &kernelNode,
                                            &memcpyNodeD2H, 1));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateHostArray(hostMem, N, 101) == true);

  // Validating hipGraphNodeGetDependencies API
  // For memcpyNodeH2D
  size_t pNumDependencies = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(memcpyNodeH2D, nullptr,
                                                &pNumDependencies));
  REQUIRE(pNumDependencies == 0);

  // For kernelNode
  pNumDependencies = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(kernelNode, nullptr,
                                                &pNumDependencies));
  REQUIRE(pNumDependencies == 1);
  hipGraphNode_t *pDependencies = reinterpret_cast<hipGraphNode_t *>(
                                  malloc(1 * sizeof(hipGraphNode_t)));

  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(kernelNode, pDependencies,
                                                &pNumDependencies));
  REQUIRE(pDependencies[0] == memcpyNodeH2D);

  // For memcpyNodeD2H
  pNumDependencies = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(memcpyNodeD2H, nullptr,
                                                &pNumDependencies));
  REQUIRE(pNumDependencies == 1);
  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(memcpyNodeD2H, pDependencies,
                                                &pNumDependencies));
  REQUIRE(pDependencies[0] == kernelNode);

  free(pDependencies);

  // Validating hipGraphNodeGetDependentNodes API
  // For memcpyNodeH2D
  size_t pNumDependentNodes = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(memcpyNodeH2D, nullptr,
                                                  &pNumDependentNodes));
  REQUIRE(pNumDependentNodes == 1);

  hipGraphNode_t *pDependentNodes = reinterpret_cast<hipGraphNode_t *>(
                                    malloc(1 * sizeof(hipGraphNode_t)));
  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(memcpyNodeH2D,
                                                  pDependentNodes,
                                                  &pNumDependentNodes));
  REQUIRE(pDependentNodes[0] == kernelNode);

  // For kernelNode
  pNumDependentNodes = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(kernelNode, nullptr,
                                                  &pNumDependentNodes));
  REQUIRE(pNumDependentNodes == 1);

  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(kernelNode, pDependentNodes,
                                                  &pNumDependentNodes));
  REQUIRE(pDependentNodes[0] == memcpyNodeD2H);

  // For memcpyNodeD2H
  pNumDependentNodes = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(memcpyNodeD2H, nullptr,
                                                  &pNumDependentNodes));
  REQUIRE(pNumDependentNodes == 0);

  free(pDependentNodes);

  // Validating hipGraphGetEdges API
  size_t numEdges = -1;

  HIP_CHECK(dyn_hipGraphGetEdges_ptr(graph, nullptr, nullptr, &numEdges));
  REQUIRE(numEdges == 2);

  hipGraphNode_t *fromnode = reinterpret_cast<hipGraphNode_t *>(
                             malloc(2 * sizeof(hipGraphNode_t)));
  hipGraphNode_t *tonode = reinterpret_cast<hipGraphNode_t *>(
                           malloc(2 * sizeof(hipGraphNode_t)));

  HIP_CHECK(dyn_hipGraphGetEdges_ptr(graph, fromnode, tonode, &numEdges));

  REQUIRE(fromnode[0] == memcpyNodeH2D);
  REQUIRE(fromnode[1] == kernelNode);
  REQUIRE(tonode[0] == kernelNode);
  REQUIRE(tonode[1] == memcpyNodeD2H);

  free(fromnode);
  free(tonode);

  // Validating hipGraphRemoveDependencies API
  HIP_CHECK(dyn_hipGraphRemoveDependencies_ptr(graph, &memcpyNodeH2D,
                                               &kernelNode, 1));

  pNumDependencies = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(memcpyNodeH2D, nullptr,
                                                &pNumDependencies));
  REQUIRE(pNumDependencies == 0);

  pNumDependencies = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(kernelNode, nullptr,
                                                &pNumDependencies));
  REQUIRE(pNumDependencies == 0);

  pNumDependencies = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependencies_ptr(memcpyNodeD2H, nullptr,
                                                &pNumDependencies));
  REQUIRE(pNumDependencies == 1);

  pNumDependentNodes = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(memcpyNodeH2D, nullptr,
                                                  &pNumDependentNodes));
  REQUIRE(pNumDependentNodes == 0);

  pNumDependentNodes = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(kernelNode, nullptr,
                                                  &pNumDependentNodes));
  REQUIRE(pNumDependentNodes == 1);

  pNumDependentNodes = -1;
  HIP_CHECK(dyn_hipGraphNodeGetDependentNodes_ptr(memcpyNodeD2H, nullptr,
                                                  &pNumDependentNodes));
  REQUIRE(pNumDependentNodes == 0);

  numEdges = -1;
  HIP_CHECK(dyn_hipGraphGetEdges_ptr(graph, nullptr, nullptr, &numEdges));
  REQUIRE(numEdges == 1);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipGraphAddEmptyNode
 *  - Graph API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_AddEmptyNode") {
  void* hipGraphAddEmptyNode_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddEmptyNode",
            &hipGraphAddEmptyNode_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddEmptyNode_ptr)(
              hipGraphNode_t *, hipGraph_t,
              const hipGraphNode_t *, size_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
                                    const hipGraphNode_t *, size_t)>
                                   (hipGraphAddEmptyNode_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 234);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t memcpyNodeH2D, emptyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memcpyNodeH2D
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  // Prepare Empty node with memcpyNodeH2D as a dependency
  ::std::vector<hipGraphNode_t> emptyNodeDependencies;
  emptyNodeDependencies.push_back(memcpyNodeH2D);

  // Validating hipGraphAddEmptyNode API
  HIP_CHECK(dyn_hipGraphAddEmptyNode_ptr(&emptyNode, graph,
                                         emptyNodeDependencies.data(),
                                         emptyNodeDependencies.size()));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem, N, 234) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipGraphChildGraphNodeGetGraph
 *  - Graph API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ChildGraphNodeGetGraph") {
  void* hipGraphChildGraphNodeGetGraph_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphChildGraphNodeGetGraph",
            &hipGraphChildGraphNodeGetGraph_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphChildGraphNodeGetGraph_ptr)(
              hipGraphNode_t, hipGraph_t *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipGraph_t *)>
                                   (hipGraphChildGraphNodeGetGraph_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem1 = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem1 != nullptr);
  fillHostArray(hostMem1, N, 100);

  int* hostMem2 = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem2 != nullptr);
  fillHostArray(hostMem2, N, 200);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t memcpyNodeH2H, memcpyNodeH2D, kernelNode, memcpyNodeD2H;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2H, graph, nullptr, 0,
            hostMem2, hostMem1, Nbytes, hipMemcpyHostToHost));

  hipGraph_t childGraph = nullptr;
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, childGraph, nullptr, 0,
            devMem, hostMem2, Nbytes, hipMemcpyHostToDevice));

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  hipKernelNodeParams kernelNodeParams{};

  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void* kernelParamArgs[2] = { reinterpret_cast<void*>(&devMem),
                               reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelParamArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, childGraph,
            kernelNodeDependencies.data(),
            kernelNodeDependencies.size(), &kernelNodeParams));

  ::std::vector<hipGraphNode_t> memcpyNodeD2HDependencies;
  memcpyNodeD2HDependencies.push_back(kernelNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeD2H, childGraph,
            memcpyNodeD2HDependencies.data(), memcpyNodeD2HDependencies.size(),
            hostMem1, devMem, Nbytes, hipMemcpyDeviceToHost));

  hipGraphNode_t childGraphNode;

  ::std::vector<hipGraphNode_t> childGraphDependencies;
  childGraphDependencies.push_back(memcpyNodeH2H);

  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph,
                                      childGraphDependencies.data(),
                                      childGraphDependencies.size(),
                                      childGraph));

  hipGraph_t embeddedChildGraph;
  HIP_CHECK(dyn_hipGraphChildGraphNodeGetGraph_ptr(childGraphNode,
                                                   &embeddedChildGraph));

  hipGraphExec_t graphExecForChildGraph;
  HIP_CHECK(hipGraphInstantiate(&graphExecForChildGraph, embeddedChildGraph,
                                nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExecForChildGraph, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateHostArray(hostMem1, N, 201) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExecForChildGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  free(hostMem1);
  free(hostMem2);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphExecChildGraphNodeSetParams API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ExecChildGraphNodeSetParams") {
  void* hipGraphExecChildGraphNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecChildGraphNodeSetParams",
            &hipGraphExecChildGraphNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphExecChildGraphNodeSetParams_ptr)(
              hipGraphExec_t, hipGraphNode_t, hipGraph_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                    hipGraph_t)>
                                   (hipGraphExecChildGraphNodeSetParams_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem1 = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem1 != nullptr);
  fillHostArray(hostMem1, N, 99);

  int* hostMem2 = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem2 != nullptr);
  fillHostArray(hostMem2, N, 100);

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, Nbytes));
  REQUIRE(devMem1 != nullptr);
  fillDeviceArray(devMem1, N, 200);

  int* devMem2 = nullptr;
  HIP_CHECK(hipMalloc(&devMem2, Nbytes));
  REQUIRE(devMem2 != nullptr);
  fillDeviceArray(devMem2, N, 300);

  hipGraphNode_t memcpyNodeH2H, memcpyNodeH2D_1, memcpyNodeH2D_2;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memcpyNodeH2H
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2H, graph, nullptr, 0,
            hostMem2, hostMem1, Nbytes, hipMemcpyHostToHost));

  hipGraph_t childGraph1 = nullptr;
  HIP_CHECK(hipGraphCreate(&childGraph1, 0));

  // Prepare memcpyNodeH2D_1
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D_1, childGraph1, nullptr, 0,
            devMem1, hostMem2, Nbytes, hipMemcpyHostToDevice));

  hipGraphNode_t childGraphNode;

  ::std::vector<hipGraphNode_t> childGraphDependencies;
  childGraphDependencies.push_back(memcpyNodeH2H);

  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph,
                                      childGraphDependencies.data(),
                                      childGraphDependencies.size(),
                                      childGraph1));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipGraph_t childGraph2 = nullptr;
  HIP_CHECK(hipGraphCreate(&childGraph2, 0));

  // Prepare memcpyNodeH2D_2
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D_2, childGraph2, nullptr, 0,
            devMem2, hostMem2, Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(dyn_hipGraphExecChildGraphNodeSetParams_ptr(graphExec,
            childGraphNode, childGraph2));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem1, N, 200) == true);
  REQUIRE(validateDeviceArray(devMem2, N, 99) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem1));
  HIP_CHECK(hipFree(devMem2));
  free(hostMem1);
  free(hostMem2);
}

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipGraphNodeSetEnabled API,
 *  - hipGraphNodeGetEnabled API from the hipGetProcAddress API
 *  - and then validates the basic functionality of those APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_SetGetEnabled") {
  void* hipGraphNodeSetEnabled_ptr = nullptr;
  void* hipGraphNodeGetEnabled_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphNodeSetEnabled",
            &hipGraphNodeSetEnabled_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphNodeGetEnabled",
            &hipGraphNodeGetEnabled_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphNodeSetEnabled_ptr)(
              hipGraphExec_t, hipGraphNode_t, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                    unsigned int)>
                                   (hipGraphNodeSetEnabled_ptr);
  hipError_t (*dyn_hipGraphNodeGetEnabled_ptr)(
              hipGraphExec_t, hipGraphNode_t, unsigned int *) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                    unsigned int *)>
                                   (hipGraphNodeGetEnabled_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 234);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 456);

  hipGraphNode_t memcpyNodeH2D;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memcpyNodeH2D
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  unsigned int isEnabled = 0;
  HIP_CHECK(dyn_hipGraphNodeGetEnabled_ptr(graphExec,
            memcpyNodeH2D, &isEnabled));
  REQUIRE(isEnabled == 1);

  unsigned int valToSet = 0;
  HIP_CHECK(dyn_hipGraphNodeSetEnabled_ptr(graphExec,
            memcpyNodeH2D, valToSet));
  HIP_CHECK(dyn_hipGraphNodeGetEnabled_ptr(graphExec,
            memcpyNodeH2D, &isEnabled));
  REQUIRE(isEnabled == 0);

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif
  REQUIRE(validateDeviceArray(devMem, N, 456) == true);

  valToSet = 1;
  HIP_CHECK(dyn_hipGraphNodeSetEnabled_ptr(graphExec,
            memcpyNodeH2D, valToSet));
  HIP_CHECK(dyn_hipGraphNodeGetEnabled_ptr(graphExec,
            memcpyNodeH2D, &isEnabled));
  REQUIRE(isEnabled == 1);

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif
  REQUIRE(validateDeviceArray(devMem, N, 234) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
}

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph APIs related to Event Record from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_EventRecord") {
  void* hipGraphAddEventRecordNode_ptr = nullptr;
  void* hipGraphEventRecordNodeGetEvent_ptr = nullptr;
  void* hipGraphEventRecordNodeSetEvent_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddEventRecordNode",
            &hipGraphAddEventRecordNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphEventRecordNodeGetEvent",
            &hipGraphEventRecordNodeGetEvent_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphEventRecordNodeSetEvent",
            &hipGraphEventRecordNodeSetEvent_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddEventRecordNode_ptr)(
              hipGraphNode_t *, hipGraph_t, const hipGraphNode_t *,
              size_t, hipEvent_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
                                    const hipGraphNode_t *, size_t,
                                    hipEvent_t)>
                                   (hipGraphAddEventRecordNode_ptr);
  hipError_t (*dyn_hipGraphEventRecordNodeGetEvent_ptr)(
              hipGraphNode_t, hipEvent_t *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipEvent_t *)>
                                   (hipGraphEventRecordNodeGetEvent_ptr);
  hipError_t (*dyn_hipGraphEventRecordNodeSetEvent_ptr)(
              hipGraphNode_t, hipEvent_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipEvent_t)>
                                   (hipGraphEventRecordNodeSetEvent_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 10);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraphNode_t startEventNode, kernelNode, stopEventNode;
  hipEvent_t resEvent;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t start = nullptr, stop = nullptr;

  HIP_CHECK(hipEventCreate(&start));
  REQUIRE(start != nullptr);

  HIP_CHECK(hipEventCreate(&stop));
  REQUIRE(stop != nullptr);

  // Preapre startEventNode
  HIP_CHECK(dyn_hipGraphAddEventRecordNode_ptr(&startEventNode, graph,
                                               nullptr, 0, start));

  // Prepare kernelNode
  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(startEventNode);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
            kernelNodeDependencies.data(), kernelNodeDependencies.size(),
            &kernelNodeParams));

  // Prepare stopEventNode
  ::std::vector<hipGraphNode_t> stopEventNodeDependencies;
  stopEventNodeDependencies.push_back(kernelNode);

  HIP_CHECK(dyn_hipGraphAddEventRecordNode_ptr(&stopEventNode, graph,
                                               nullptr, 0, stop));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventSynchronize(stop));

  REQUIRE(validateDeviceArray(devMem, N, 11) == true);

  float time = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&time, start, stop));
  REQUIRE(time > 0.0f);

  HIP_CHECK(dyn_hipGraphEventRecordNodeGetEvent_ptr(
            startEventNode, &resEvent));
  REQUIRE(resEvent == start);

  HIP_CHECK(dyn_hipGraphEventRecordNodeGetEvent_ptr(
            stopEventNode, &resEvent));
  REQUIRE(resEvent == stop);

  hipEvent_t newStart = nullptr, newStop = nullptr;

  HIP_CHECK(hipEventCreate(&newStart));
  REQUIRE(newStart != nullptr);

  HIP_CHECK(hipEventCreate(&newStop));
  REQUIRE(newStop != nullptr);

  HIP_CHECK(dyn_hipGraphEventRecordNodeSetEvent_ptr(
            startEventNode, newStart));
  HIP_CHECK(dyn_hipGraphEventRecordNodeSetEvent_ptr(
            stopEventNode, newStop));

  HIP_CHECK(dyn_hipGraphEventRecordNodeGetEvent_ptr(
            startEventNode, &resEvent));
  REQUIRE(resEvent == newStart);

  HIP_CHECK(dyn_hipGraphEventRecordNodeGetEvent_ptr(
            stopEventNode, &resEvent));
  REQUIRE(resEvent == newStop);

  hipGraphExec_t newGraphExec;
  HIP_CHECK(hipGraphInstantiate(&newGraphExec, graph,
                                nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(newGraphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventSynchronize(newStop));

  REQUIRE(validateDeviceArray(devMem, N, 12) == true);

  float newTime = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&newTime, newStart, newStop));
  REQUIRE(newTime > 0.0f);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  HIP_CHECK(hipGraphExecDestroy(newGraphExec));
  HIP_CHECK(hipEventDestroy(newStart));
  HIP_CHECK(hipEventDestroy(newStop));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphExecEventRecordNodeSetEvent API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ExecEventRecordSetEvent") {
  void* hipGraphExecEventRecordNodeSetEvent_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecEventRecordNodeSetEvent",
            &hipGraphExecEventRecordNodeSetEvent_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphExecEventRecordNodeSetEvent_ptr)(
              hipGraphExec_t, hipGraphNode_t, hipEvent_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                    hipEvent_t)>
                                   (hipGraphExecEventRecordNodeSetEvent_ptr);
  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 10);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraphNode_t startEventNode, kernelNode, stopEventNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t start = nullptr, stop = nullptr, stopNew = nullptr;

  HIP_CHECK(hipEventCreate(&start));
  REQUIRE(start != nullptr);

  HIP_CHECK(hipEventCreate(&stop));
  REQUIRE(stop != nullptr);

  HIP_CHECK(hipEventCreate(&stopNew));
  REQUIRE(stopNew != nullptr);

  // Preapre startEventNode
  HIP_CHECK(hipGraphAddEventRecordNode(&startEventNode, graph,
                                       nullptr, 0, start));

  // Prepare kernelNode
  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(startEventNode);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
            kernelNodeDependencies.data(), kernelNodeDependencies.size(),
            &kernelNodeParams));

  // Prepare stopEventNode
  ::std::vector<hipGraphNode_t> stopEventNodeDependencies;
  stopEventNodeDependencies.push_back(kernelNode);

  HIP_CHECK(hipGraphAddEventRecordNode(&stopEventNode, graph,
                                       nullptr, 0, stop));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(dyn_hipGraphExecEventRecordNodeSetEvent_ptr(graphExec,
                                                        stopEventNode,
                                                        stopNew));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventSynchronize(stopNew));

  REQUIRE(validateDeviceArray(devMem, N, 11) == true);

  float time = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&time, start, stopNew));
  REQUIRE(time > 0.0f);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipEventDestroy(stopNew));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
}

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph APIs related to Event Wait from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_EventWait") {
  void* hipGraphAddEventWaitNode_ptr = nullptr;
  void* hipGraphEventWaitNodeGetEvent_ptr = nullptr;
  void* hipGraphEventWaitNodeSetEvent_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddEventWaitNode",
            &hipGraphAddEventWaitNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphEventWaitNodeGetEvent",
            &hipGraphEventWaitNodeGetEvent_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphEventWaitNodeSetEvent",
            &hipGraphEventWaitNodeSetEvent_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddEventWaitNode_ptr)(
              hipGraphNode_t *, hipGraph_t,
              const hipGraphNode_t *, size_t, hipEvent_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
              const hipGraphNode_t *, size_t, hipEvent_t)>
                                   (hipGraphAddEventWaitNode_ptr);
  hipError_t (*dyn_hipGraphEventWaitNodeGetEvent_ptr)(
              hipGraphNode_t, hipEvent_t *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipEvent_t *)>
                                   (hipGraphEventWaitNodeGetEvent_ptr);
  hipError_t (*dyn_hipGraphEventWaitNodeSetEvent_ptr)(
              hipGraphNode_t, hipEvent_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipEvent_t)>
                                   (hipGraphEventWaitNodeSetEvent_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t recordEventNode, memcpyNodeH2DNode, waitEventNode, kernelNode;

  hipEvent_t waitEvent = nullptr;
  HIP_CHECK(hipEventCreate(&waitEvent));
  REQUIRE(waitEvent != nullptr);

  hipGraph_t graph1 = nullptr;
  HIP_CHECK(hipGraphCreate(&graph1, 0));

  hipStream_t stream1;
  HIP_CHECK(hipStreamCreate(&stream1));

  // Prepare graph1 with memcpyNodeH2DNode and recordEventNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2DNode, graph1, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  ::std::vector<hipGraphNode_t> recordEventNodeDependencies;
  recordEventNodeDependencies.push_back(memcpyNodeH2DNode);

  HIP_CHECK(hipGraphAddEventRecordNode(&recordEventNode, graph1,
                                       recordEventNodeDependencies.data(),
                                       recordEventNodeDependencies.size(),
                                       waitEvent));

  hipGraph_t graph2 = nullptr;
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  hipStream_t stream2;
  HIP_CHECK(hipStreamCreate(&stream2));

  // Prepare graph2 with memcpyNodeH2DNode and recordEventNode
  HIP_CHECK(dyn_hipGraphAddEventWaitNode_ptr(&waitEventNode, graph2,
                                             nullptr, 0, waitEvent));

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(waitEventNode);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addTwoKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph2,
            kernelNodeDependencies.data(), kernelNodeDependencies.size(),
            &kernelNodeParams));

  // Execute graph1 on stream 1, graph2 on stream 2
  hipGraphExec_t graphExec1, graphExec2;
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
  HIP_CHECK(hipGraphLaunch(graphExec2, stream2));

  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamSynchronize(stream2));

  REQUIRE(validateDeviceArray(devMem, N, 12) == true);

  HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

  hipEvent_t resEvent;
  HIP_CHECK(dyn_hipGraphEventWaitNodeGetEvent_ptr(
            waitEventNode, &resEvent));
  REQUIRE(resEvent == waitEvent);

  hipEvent_t newWaitEvent = nullptr;

  HIP_CHECK(hipEventCreate(&newWaitEvent));
  REQUIRE(newWaitEvent != nullptr);

  HIP_CHECK(hipGraphEventRecordNodeSetEvent(
            recordEventNode, newWaitEvent));

  HIP_CHECK(dyn_hipGraphEventWaitNodeSetEvent_ptr(
            waitEventNode, newWaitEvent));

  HIP_CHECK(dyn_hipGraphEventWaitNodeGetEvent_ptr(
            waitEventNode, &resEvent));
  REQUIRE(resEvent == newWaitEvent);

  // Execute graph1 on stream 1, graph2 on stream 2
  hipGraphExec_t newGraphExec1, newGraphExec2;
  HIP_CHECK(hipGraphInstantiate(&newGraphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&newGraphExec2, graph2, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(newGraphExec1, stream1));
  HIP_CHECK(hipGraphLaunch(newGraphExec2, stream2));

  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamSynchronize(stream2));

  REQUIRE(validateDeviceArray(devMem, N, 14) == true);

  HIP_CHECK(hipEventDestroy(waitEvent));
  HIP_CHECK(hipEventDestroy(newWaitEvent));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(newGraphExec1));
  HIP_CHECK(hipGraphExecDestroy(newGraphExec2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  free(hostMem);
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphExecEventWaitNodeSetEvent API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ExecEventWaitSetEvent") {
  void* hipGraphExecEventWaitNodeSetEvent_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecEventWaitNodeSetEvent",
            &hipGraphExecEventWaitNodeSetEvent_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphExecEventWaitNodeSetEvent_ptr)(
              hipGraphExec_t, hipGraphNode_t, hipEvent_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                    hipEvent_t)>
                                   (hipGraphExecEventWaitNodeSetEvent_ptr);

  int N = 1024 * 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t recordEventNode, memcpyNodeH2DNode, waitEventNode, kernelNode;

  hipEvent_t waitEvent1 = nullptr;
  HIP_CHECK(hipEventCreate(&waitEvent1));
  REQUIRE(waitEvent1 != nullptr);
  hipEvent_t waitEvent2 = nullptr;
  HIP_CHECK(hipEventCreate(&waitEvent2));
  REQUIRE(waitEvent2 != nullptr);

  hipGraph_t graph1 = nullptr;
  HIP_CHECK(hipGraphCreate(&graph1, 0));

  hipStream_t stream1;
  HIP_CHECK(hipStreamCreate(&stream1));

  // Prepare graph1 with memcpyNodeH2DNode and recordEventNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2DNode, graph1, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  ::std::vector<hipGraphNode_t> recordEventNodeDependencies;
  recordEventNodeDependencies.push_back(memcpyNodeH2DNode);

  HIP_CHECK(hipGraphAddEventRecordNode(&recordEventNode, graph1,
                                       recordEventNodeDependencies.data(),
                                       recordEventNodeDependencies.size(),
                                       waitEvent1));

  hipGraph_t graph2 = nullptr;
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  hipStream_t stream2;
  HIP_CHECK(hipStreamCreate(&stream2));

  // Prepare graph2 with memcpyNodeH2DNode and recordEventNode
  HIP_CHECK(hipGraphAddEventWaitNode(&waitEventNode, graph2,
                                             nullptr, 0, waitEvent2));

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(waitEventNode);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addTwoKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph2,
            kernelNodeDependencies.data(), kernelNodeDependencies.size(),
            &kernelNodeParams));

  // Execute graph1 on stream 1, graph2 on stream 2
  hipGraphExec_t graphExec1, graphExec2;
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));

  // set waitEvent1 for waitEventNode in graphExec2
  HIP_CHECK(dyn_hipGraphExecEventWaitNodeSetEvent_ptr(graphExec2,
            waitEventNode, waitEvent1));

  HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
  HIP_CHECK(hipGraphLaunch(graphExec2, stream2));

  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamSynchronize(stream2));

  REQUIRE(validateDeviceArray(devMem, N, 12) == true);

  HIP_CHECK(hipEventDestroy(waitEvent1));
  HIP_CHECK(hipEventDestroy(waitEvent2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  free(hostMem);
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphUpload API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_hipGraphUpload") {
  void* hipGraphUpload_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphUpload",
            &hipGraphUpload_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphUpload_ptr)(
              hipGraphExec_t, hipStream_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipStream_t)>
                                   (hipGraphUpload_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 10);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraphNode_t kernelNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare kernelNode
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
                                  nullptr, 0,
                                  &kernelNodeParams));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(dyn_hipGraphUpload_ptr(graphExec, stream));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));


  REQUIRE(validateDeviceArray(devMem, N, 11) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphKernelNodeCopyAttributes API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_KernelNodeCopyAttributes") {
  void* hipGraphKernelNodeCopyAttributes_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphKernelNodeCopyAttributes",
            &hipGraphKernelNodeCopyAttributes_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphKernelNodeCopyAttributes_ptr)(
              hipGraphNode_t, hipGraphNode_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, hipGraphNode_t)>
                                   (hipGraphKernelNodeCopyAttributes_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 234);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraphNode_t kernelNode1, kernelNode2;
  hipKernelNodeAttrValue attributeToSet, attributeToGet;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare kernelNode1
  hipKernelNodeParams kernelNodeParams1{};
  kernelNodeParams1.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams1.gridDim = dim3(1, 1, 1);
  kernelNodeParams1.blockDim = dim3(1, 1, 1);
  kernelNodeParams1.sharedMemBytes = 0;

  void* kernelArgs1[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams1.kernelParams = kernelArgs1;
  kernelNodeParams1.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode1, graph,
                                  nullptr, 0,
                                  &kernelNodeParams1));

  // Prepare kernelNode2
  hipKernelNodeParams kernelNodeParams2{};
  kernelNodeParams2.func = reinterpret_cast<void*>(addTwoKernel);
  kernelNodeParams2.gridDim = dim3(1, 1, 1);
  kernelNodeParams2.blockDim = dim3(1, 1, 1);
  kernelNodeParams2.sharedMemBytes = 0;

  void* kernelArgs2[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams2.kernelParams = kernelArgs2;
  kernelNodeParams2.extra = nullptr;

  ::std::vector<hipGraphNode_t> kernelNode2Dependencies;
  kernelNode2Dependencies.push_back(kernelNode1);

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode2, graph,
                                  kernelNode2Dependencies.data(),
                                  kernelNode2Dependencies.size(),
                                  &kernelNodeParams2));

  attributeToSet.cooperative = 1;
  HIP_CHECK(hipGraphKernelNodeSetAttribute(kernelNode1,
            hipKernelNodeAttributeCooperative, &attributeToSet));

  HIP_CHECK(hipGraphKernelNodeGetAttribute(kernelNode1,
            hipKernelNodeAttributeCooperative, &attributeToGet));
  REQUIRE(attributeToGet.cooperative == 1);

  HIP_CHECK(hipGraphKernelNodeGetAttribute(kernelNode2,
            hipKernelNodeAttributeCooperative, &attributeToGet));
  REQUIRE(attributeToGet.cooperative == 0);

  HIP_CHECK(dyn_hipGraphKernelNodeCopyAttributes_ptr(kernelNode1,
                                                     kernelNode2));

  HIP_CHECK(hipGraphKernelNodeGetAttribute(kernelNode2,
            hipKernelNodeAttributeCooperative, &attributeToGet));
  REQUIRE(attributeToGet.cooperative == 1);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(validateDeviceArray(devMem, N, 237) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
}

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipDrvGraphAddMemsetNode API,
 *  - hipDrvGraphAddMemcpyNode API from the hipGetProcAddress API
 *  - and then validates the basic functionality of those APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_DrvAddMemsetAddMemcpy") {
  CHECK_IMAGE_SUPPORT

  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  void* hipDrvGraphAddMemsetNode_ptr = nullptr;
  void* hipDrvGraphAddMemcpyNode_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipDrvGraphAddMemsetNode",
            &hipDrvGraphAddMemsetNode_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipDrvGraphAddMemcpyNode",
            &hipDrvGraphAddMemcpyNode_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipDrvGraphAddMemsetNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const HIP_MEMSET_NODE_PARAMS *, hipCtx_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const HIP_MEMSET_NODE_PARAMS *, hipCtx_t)>
    (hipDrvGraphAddMemsetNode_ptr);

  hipError_t (*dyn_hipDrvGraphAddMemcpyNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const HIP_MEMCPY3D *, hipCtx_t) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const HIP_MEMCPY3D *, hipCtx_t)>
    (hipDrvGraphAddMemcpyNode_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char *devMemSrc = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc),
                             &pitch, width, height));
  REQUIRE(devMemSrc != nullptr);

  char* hostMemDst = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst != nullptr);

  hipGraphNode_t memsetNode, memcpyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Validating hipDrvGraphAddMemsetNode API
  // Prepare memsetNode
  HIP_MEMSET_NODE_PARAMS memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  memsetParams.value = value;
  memsetParams.pitch = pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = width;
  memsetParams.height = height;

  HIP_CHECK(dyn_hipDrvGraphAddMemsetNode_ptr(&memsetNode, graph,
                                             nullptr, 0,
                                             &memsetParams, context));

  // Validating hipDrvGraphAddMemcpyNode API
  // Prepare memcpyNode with memsetNode as a dependency
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  HIP_MEMCPY3D memcpyParams{};
  memcpyParams.srcMemoryType = hipMemoryTypeDevice;
  memcpyParams.srcDevice = reinterpret_cast<hipDeviceptr_t>(devMemSrc);
  memcpyParams.srcPitch = pitch;
  memcpyParams.dstMemoryType = hipMemoryTypeHost;
  memcpyParams.dstHost = hostMemDst;
  memcpyParams.dstPitch = width;
  memcpyParams.srcXInBytes = 0;
  memcpyParams.srcY = 0;
  memcpyParams.srcZ = 0;
  memcpyParams.dstXInBytes = 0;
  memcpyParams.dstY = 0;
  memcpyParams.dstZ = 0;
  memcpyParams.WidthInBytes = width;
  memcpyParams.Height = height;
  memcpyParams.Depth = 1;

  HIP_CHECK(dyn_hipDrvGraphAddMemcpyNode_ptr(&memcpyNode, graph,
                                             memcpyNodeDependencies.data(),
                                             memcpyNodeDependencies.size(),
                                             &memcpyParams, context));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateArrayT<char>(hostMemDst, N, value) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMemSrc));
  free(hostMemDst);

  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipDrvGraphExecMemcpyNodeSetParams API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_DrvGraphExecMemcpyNodeSetParams") {
  CHECK_IMAGE_SUPPORT

  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  void* hipDrvGraphExecMemcpyNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipDrvGraphExecMemcpyNodeSetParams",
            &hipDrvGraphExecMemcpyNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipDrvGraphExecMemcpyNodeSetParams_ptr)(hipGraphExec_t,
              hipGraphNode_t, const HIP_MEMCPY3D*, hipCtx_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                 const HIP_MEMCPY3D*, hipCtx_t)>
    (hipDrvGraphExecMemcpyNodeSetParams_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 50;
  size_t pitch;

  char *devMemSrc = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc),
                             &pitch, width, height));
  REQUIRE(devMemSrc != nullptr);

  char* hostMemDst1 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst1 != nullptr);
  fillCharHostArray(hostMemDst1, N, 100);

  char* hostMemDst2 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst2 != nullptr);
  fillCharHostArray(hostMemDst2, N, 110);

  hipGraphNode_t memsetNode, memcpyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memset node
  HIP_MEMSET_NODE_PARAMS initialMemsetParams{};
  initialMemsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  initialMemsetParams.value = value;
  initialMemsetParams.pitch = pitch;
  initialMemsetParams.elementSize = sizeof(char);
  initialMemsetParams.width = width;
  initialMemsetParams.height = height;

  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph,
            nullptr, 0, &initialMemsetParams, context));

  // Prepare memcpyNode
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  HIP_MEMCPY3D initialParms{};
  initialParms.srcMemoryType = hipMemoryTypeDevice;
  initialParms.srcDevice = reinterpret_cast<hipDeviceptr_t>(devMemSrc);
  initialParms.srcPitch = pitch;
  initialParms.dstMemoryType = hipMemoryTypeHost;
  initialParms.dstHost = hostMemDst1;
  initialParms.dstPitch = width;
  initialParms.srcXInBytes = 0;
  initialParms.srcY = 0;
  initialParms.srcZ = 0;
  initialParms.dstXInBytes = 0;
  initialParms.dstY = 0;
  initialParms.dstZ = 0;
  initialParms.WidthInBytes = width;
  initialParms.Height = height;
  initialParms.Depth = 1;

  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph,
            memcpyNodeDependencies.data(),
            memcpyNodeDependencies.size(), &initialParms, context));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateArrayT<char>(hostMemDst1, N, 50) == true);
  REQUIRE(validateArrayT<char>(hostMemDst2, N, 110) == true);

  HIP_MEMCPY3D newMemcpyParms{};
  newMemcpyParms.srcMemoryType = hipMemoryTypeDevice;
  newMemcpyParms.srcDevice = reinterpret_cast<hipDeviceptr_t>(devMemSrc);
  newMemcpyParms.srcPitch = pitch;
  newMemcpyParms.dstMemoryType = hipMemoryTypeHost;
  newMemcpyParms.dstHost = hostMemDst2;
  newMemcpyParms.dstPitch = width;
  newMemcpyParms.srcXInBytes = 0;
  newMemcpyParms.srcY = 0;
  newMemcpyParms.srcZ = 0;
  newMemcpyParms.dstXInBytes = 0;
  newMemcpyParms.dstY = 0;
  newMemcpyParms.dstZ = 0;
  newMemcpyParms.WidthInBytes = width;
  newMemcpyParms.Height = height;
  newMemcpyParms.Depth = 1;

  // Validating hipDrvGraphExecMemcpyNodeSetParams API
  HIP_CHECK(dyn_hipDrvGraphExecMemcpyNodeSetParams_ptr(graphExec, memcpyNode,
                                               &newMemcpyParms, context));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateArrayT<char>(hostMemDst1, N, 50) == true);
  REQUIRE(validateArrayT<char>(hostMemDst2, N, 50) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMemSrc));
  free(hostMemDst1);
  free(hostMemDst2);

  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipDrvGraphAddMemFreeNode API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_hipDrvGraphAddMemFreeNode") {
  void* hipDrvGraphAddMemFreeNode_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipDrvGraphAddMemFreeNode",
            &hipDrvGraphAddMemFreeNode_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipDrvGraphAddMemFreeNode_ptr)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, void *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, void *)>
    (hipDrvGraphAddMemFreeNode_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int *devMem = nullptr;
  int *devMemToCheck = nullptr;

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memAllocNode, memFreeNode;

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
  HIP_CHECK(hipGraphAddMemAllocNode(&memAllocNode, graph,
                                    nullptr, 0, &memAllocNodeParams));
  devMem = reinterpret_cast<int*>(memAllocNodeParams.dptr);
  devMemToCheck = reinterpret_cast<int*>(memAllocNodeParams.dptr);

  ::std::vector<hipGraphNode_t> memFreeNodeDependencies;
  memFreeNodeDependencies.push_back(memAllocNode);

  // Validating hipDrvGraphAddMemFreeNode API
  HIP_CHECK(dyn_hipDrvGraphAddMemFreeNode_ptr(&memFreeNode, graph,
            memFreeNodeDependencies.data(), memFreeNodeDependencies.size(),
            reinterpret_cast<void*>(devMem)));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(devMemToCheck != nullptr);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphInstantiateWithFlags API and hipGraphExecGetFlags API
 *  - from the hipGetProcAddress API and then validates the basic
 *  - functionality of those APIs using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_InstantiateWithFlagsAndGetFlags") {
  void* hipGraphInstantiateWithFlags_ptr = nullptr;
  void* hipGraphExecGetFlags_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphInstantiateWithFlags",
            &hipGraphInstantiateWithFlags_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecGetFlags",
            &hipGraphExecGetFlags_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphInstantiateWithFlags_ptr)(hipGraphExec_t *,
                                                     hipGraph_t,
                                                     uint64_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t *,
                                    hipGraph_t,
                                    uint64_t)>
                                    (hipGraphInstantiateWithFlags_ptr);

  hipError_t (*dyn_hipGraphExecGetFlags_ptr)(hipGraphExec_t, uint64_t*) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, uint64_t*)>
    (hipGraphExecGetFlags_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 234);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraphNode_t kernelNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare kernelNode
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
                                  nullptr, 0,
                                  &kernelNodeParams));

  uint64_t flags = GENERATE(hipGraphInstantiateFlagAutoFreeOnLaunch,
                            hipGraphInstantiateFlagUseNodePriority);

  hipGraphExec_t graphExec;
  HIP_CHECK(dyn_hipGraphInstantiateWithFlags_ptr(&graphExec, graph, flags));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(validateDeviceArray(devMem, N, 235) == true);

  uint64_t receivedFlags;
  HIP_CHECK(dyn_hipGraphExecGetFlags_ptr(graphExec, &receivedFlags));
  REQUIRE(receivedFlags == flags);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - User Object related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_UserObject") {
  void* hipUserObjectCreate_ptr = nullptr;
  void* hipUserObjectRetain_ptr = nullptr;
  void* hipUserObjectRelease_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipUserObjectCreate",
            &hipUserObjectCreate_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipUserObjectRetain",
            &hipUserObjectRetain_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipUserObjectRelease",
            &hipUserObjectRelease_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipUserObjectCreate_ptr)(
              hipUserObject_t *, void *, hipHostFn_t,
              unsigned int, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipUserObject_t *, void *, hipHostFn_t,
                                    unsigned int, unsigned int)>
                                   (hipUserObjectCreate_ptr);
  hipError_t (*dyn_hipUserObjectRetain_ptr)(
              hipUserObject_t, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipUserObject_t, unsigned int)>
                                   (hipUserObjectRetain_ptr);
  hipError_t (*dyn_hipUserObjectRelease_ptr)(
              hipUserObject_t, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipUserObject_t, unsigned int)>
                                   (hipUserObjectRelease_ptr);

  int* object = new int();
  REQUIRE(object != nullptr);

  int initialRefCount = 3;
  int retainRefCount = 4;

  hipUserObject_t userObject;

  HIP_CHECK(dyn_hipUserObjectCreate_ptr(&userObject, object, destroyIntObject,
            initialRefCount, hipUserObjectNoDestructorSync));
  REQUIRE(userObject != nullptr);

  HIP_CHECK(dyn_hipUserObjectRetain_ptr(userObject, retainRefCount));

  HIP_CHECK(dyn_hipUserObjectRelease_ptr(userObject,
            initialRefCount + retainRefCount));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph APIs related to Symbol from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_Symbol") {
  CHECK_IMAGE_SUPPORT

  void* hipGraphAddMemcpyNodeToSymbol_ptr = nullptr;
  void* hipGraphAddMemcpyNodeFromSymbol_ptr = nullptr;
  void* hipGraphMemcpyNodeSetParamsToSymbol_ptr = nullptr;
  void* hipGraphMemcpyNodeSetParamsFromSymbol_ptr = nullptr;
  void* hipGraphExecMemcpyNodeSetParamsToSymbol_ptr = nullptr;
  void* hipGraphExecMemcpyNodeSetParamsFromSymbol_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddMemcpyNodeToSymbol",
            &hipGraphAddMemcpyNodeToSymbol_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphAddMemcpyNodeFromSymbol",
            &hipGraphAddMemcpyNodeFromSymbol_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemcpyNodeSetParamsToSymbol",
            &hipGraphMemcpyNodeSetParamsToSymbol_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphMemcpyNodeSetParamsFromSymbol",
            &hipGraphMemcpyNodeSetParamsFromSymbol_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecMemcpyNodeSetParamsToSymbol",
            &hipGraphExecMemcpyNodeSetParamsToSymbol_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecMemcpyNodeSetParamsFromSymbol",
            &hipGraphExecMemcpyNodeSetParamsFromSymbol_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphAddMemcpyNodeToSymbol_ptr)(hipGraphNode_t *,
    hipGraph_t, const hipGraphNode_t *, size_t, const void *, const void *,
    size_t, size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, const void *, const void *,
    size_t, size_t, hipMemcpyKind)>
    (hipGraphAddMemcpyNodeToSymbol_ptr);

  hipError_t (*dyn_hipGraphAddMemcpyNodeFromSymbol_ptr)(hipGraphNode_t *,
    hipGraph_t, const hipGraphNode_t *, size_t, void *, const void *,
    size_t, size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t *, hipGraph_t,
    const hipGraphNode_t *, size_t, void *, const void *,
    size_t, size_t, hipMemcpyKind)>
    (hipGraphAddMemcpyNodeFromSymbol_ptr);

  hipError_t (*dyn_hipGraphMemcpyNodeSetParamsToSymbol_ptr)(hipGraphNode_t,
    const void *, const void *, size_t, size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t,
    const void *, const void *, size_t, size_t, hipMemcpyKind)>
    (hipGraphMemcpyNodeSetParamsToSymbol_ptr);

  hipError_t (*dyn_hipGraphMemcpyNodeSetParamsFromSymbol_ptr)(hipGraphNode_t ,
    void *, const void *, size_t, size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t ,
    void *, const void *, size_t, size_t, hipMemcpyKind)>
    (hipGraphMemcpyNodeSetParamsFromSymbol_ptr);

  hipError_t (*dyn_hipGraphExecMemcpyNodeSetParamsToSymbol_ptr)(hipGraphExec_t,
    hipGraphNode_t, const void *, const void *, size_t,
    size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t,
    hipGraphNode_t, const void *, const void *, size_t,
    size_t, hipMemcpyKind)>
    (hipGraphExecMemcpyNodeSetParamsToSymbol_ptr);

  hipError_t (*dyn_hipGraphExecMemcpyNodeSetParamsFromSymbol_ptr)(
    hipGraphExec_t, hipGraphNode_t, void *, const void *,
    size_t, size_t, hipMemcpyKind) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t,
    hipGraphNode_t, void *, const void *, size_t, size_t, hipMemcpyKind)>
    (hipGraphExecMemcpyNodeSetParamsFromSymbol_ptr);

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, Nbytes));
  REQUIRE(devMem1 != nullptr);
  fillDeviceArray(devMem1, N, 100);

  int* devMem2 = nullptr;
  HIP_CHECK(hipMalloc(&devMem2, Nbytes));
  REQUIRE(devMem2 != nullptr);
  fillDeviceArray(devMem2, N, 0);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode;

  HIP_CHECK(dyn_hipGraphAddMemcpyNodeToSymbol_ptr(&memcpyToSymbolNode, graph,
                                                  nullptr,
                                                  0,
                                                  HIP_SYMBOL(symbolData1),
                                                  devMem1, Nbytes, 0,
                                                  hipMemcpyDeviceToDevice));

  ::std::vector<hipGraphNode_t> memcpyFromSymbolNodeDeps;
  memcpyFromSymbolNodeDeps.push_back(memcpyToSymbolNode);

  HIP_CHECK(dyn_hipGraphAddMemcpyNodeFromSymbol_ptr(
            &memcpyFromSymbolNode, graph,
            memcpyFromSymbolNodeDeps.data(),
            memcpyFromSymbolNodeDeps.size(),
            devMem2,
            HIP_SYMBOL(symbolData1),
            Nbytes, 0,
            hipMemcpyDeviceToDevice));

  hipGraphExec_t graphExec1;
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec1, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem2, N, 100));

  fillDeviceArray(devMem2, N, 0);
  HIP_CHECK(dyn_hipGraphMemcpyNodeSetParamsToSymbol_ptr(
            memcpyToSymbolNode,
            HIP_SYMBOL(symbolData2),
            devMem1, Nbytes, 0,
            hipMemcpyDeviceToDevice));
  HIP_CHECK(dyn_hipGraphMemcpyNodeSetParamsFromSymbol_ptr(
            memcpyFromSymbolNode, devMem2,
            HIP_SYMBOL(symbolData2),
            Nbytes, 0,
            hipMemcpyDeviceToDevice));

  hipGraphExec_t graphExec2;
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec2, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem2, N, 100));

  fillDeviceArray(devMem2, N, 0);

  HIP_CHECK(dyn_hipGraphExecMemcpyNodeSetParamsToSymbol_ptr(
            graphExec2, memcpyToSymbolNode,
            HIP_SYMBOL(symbolData3),
            devMem1,
            Nbytes, 0,
            hipMemcpyDeviceToDevice));
  HIP_CHECK(dyn_hipGraphExecMemcpyNodeSetParamsFromSymbol_ptr(
            graphExec2, memcpyFromSymbolNode,
            devMem2,
            HIP_SYMBOL(symbolData3),
            Nbytes, 0,
            hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphLaunch(graphExec2, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem2, N, 100));

  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem1));
  HIP_CHECK(hipFree(devMem2));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph APIs related to Stream Capture Info from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_StreamGetCaptureInfo") {
  void* hipStreamGetCaptureInfo_ptr = nullptr;
  void* hipStreamGetCaptureInfo_v2_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipStreamGetCaptureInfo",
            &hipStreamGetCaptureInfo_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipStreamGetCaptureInfo_v2",
            &hipStreamGetCaptureInfo_v2_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipStreamGetCaptureInfo_ptr)(
              hipStream_t, hipStreamCaptureStatus *,
              unsigned long long *) =  // NOLINT
    reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureStatus *,
                                    unsigned long long *)>  // NOLINT
                                   (hipStreamGetCaptureInfo_ptr);

  hipError_t (*dyn_hipStreamGetCaptureInfo_v2_ptr)(
              hipStream_t, hipStreamCaptureStatus *,
              unsigned long long *,  // NOLINT
              hipGraph_t *, const hipGraphNode_t **, size_t *) =
    reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureStatus *,
                                    unsigned long long *,  // NOLINT
                                    hipGraph_t *,
                                    const hipGraphNode_t **, size_t *)>
                                    (hipStreamGetCaptureInfo_v2_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamBeginCapture(stream,
                                  hipStreamCaptureModeGlobal));

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes,
                           hipMemcpyHostToDevice, stream));
  addOneKernel<<< 1, 1, 0, stream >>>(devMem , N);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  // Validating hipStreamGetCaptureInfo API
  hipStreamCaptureStatus
  pCaptureStatusWithOrgApi = hipStreamCaptureStatusNone,
  pCaptureStatusWithFuncPtr = hipStreamCaptureStatusNone;

  unsigned long long pIdWithOrgApi = 0, pIdWithFuncPtr = 0;  // NOLINT

  HIP_CHECK(hipStreamGetCaptureInfo(stream,
            &pCaptureStatusWithOrgApi, &pIdWithOrgApi));
  HIP_CHECK(dyn_hipStreamGetCaptureInfo_ptr(stream,
            &pCaptureStatusWithFuncPtr, &pIdWithFuncPtr));

  REQUIRE(pCaptureStatusWithFuncPtr == pCaptureStatusWithOrgApi);
  REQUIRE(pIdWithFuncPtr == pIdWithOrgApi);

  // Validating hipStreamGetCaptureInfo_v2 API
  hipStreamCaptureStatus captureStatus_out_org, captureStatus_out_ptr;
  unsigned long long id_out_org, id_out_ptr;  // NOLINT
  hipGraph_t graph_out_org, graph_out_ptr;
  const hipGraphNode_t *dependencies_out_org{};
  const hipGraphNode_t *dependencies_out_ptr{};
  size_t numDependencies_out_org, numDependencies_out_ptr;

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus_out_org,
            &id_out_org, &graph_out_org,
            &dependencies_out_org, &numDependencies_out_org));
  HIP_CHECK(dyn_hipStreamGetCaptureInfo_v2_ptr(stream,
            &captureStatus_out_ptr, &id_out_ptr, &graph_out_ptr,
            &dependencies_out_ptr, &numDependencies_out_ptr));

  REQUIRE(captureStatus_out_ptr == captureStatus_out_org);
  REQUIRE(id_out_ptr == id_out_org);
  REQUIRE(graph_out_ptr == graph_out_org);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(validateHostArray(hostMem, N, 11) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  free(hostMem);
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipStreamUpdateCaptureDependencies API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_hipStreamUpdateCaptureDeps") {
  void* hipStreamUpdateCaptureDependencies_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipStreamUpdateCaptureDependencies",
            &hipStreamUpdateCaptureDependencies_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipStreamUpdateCaptureDependencies_ptr)(hipStream_t,
    hipGraphNode_t *, size_t, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipStream_t,
    hipGraphNode_t *, size_t, unsigned int)>
    (hipStreamUpdateCaptureDependencies_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 100);

  int* devMemNew = nullptr;
  HIP_CHECK(hipMalloc(&devMemNew, Nbytes));
  REQUIRE(devMemNew != nullptr);
  fillDeviceArray(devMemNew, N, 1);

  hipGraphNode_t kernelNode, kernelNodeNew;

  hipGraph_t graph = nullptr;
  hipGraphExec_t graphExec;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamBeginCapture(stream,
                                  hipStreamCaptureModeGlobal));

  hipStreamCaptureStatus captureStatus_out_org;
  unsigned long long id_out_org;  // NOLINT
  hipGraph_t graph_out_org;
  const hipGraphNode_t *dependencies_out_org{};
  size_t numDependencies_out_org;

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus_out_org,
                                       &id_out_org, &graph_out_org,
                                       &dependencies_out_org,
                                       &numDependencies_out_org));

  REQUIRE(numDependencies_out_org == 0);

  SECTION("With_Flag_hipStreamAddCaptureDependencies") {
    hipKernelNodeParams kernelNodeParams{};
    kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(1, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;

    void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                            reinterpret_cast<void*>(&N) };
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph_out_org,
                                    nullptr, 0,
                                    &kernelNodeParams));

    HIP_CHECK(dyn_hipStreamUpdateCaptureDependencies_ptr(
              stream, &kernelNode, 1, hipStreamAddCaptureDependencies));

    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus_out_org,
                                         &id_out_org, &graph_out_org,
                                         &dependencies_out_org,
                                         &numDependencies_out_org));

    REQUIRE(numDependencies_out_org == 1);
    REQUIRE(dependencies_out_org[0] == kernelNode);

    HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes,
                             hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    #ifdef _WIN32
    HIP_CHECK(hipStreamSynchronize(stream));
    #endif

    REQUIRE(validateHostArray(hostMem, N, 101) == true);
  }

  SECTION("With_Flag_hipStreamSetCaptureDependencies") {
    hipKernelNodeParams kernelNodeParams{};
    kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(1, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;

    void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                            reinterpret_cast<void*>(&N) };
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph_out_org,
                                    nullptr, 0,
                                    &kernelNodeParams));

    HIP_CHECK(dyn_hipStreamUpdateCaptureDependencies_ptr(
              stream, &kernelNode, 1, hipStreamAddCaptureDependencies));

    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus_out_org,
                                         &id_out_org, &graph_out_org,
                                         &dependencies_out_org,
                                         &numDependencies_out_org));
    REQUIRE(numDependencies_out_org == 1);
    REQUIRE(dependencies_out_org[0] == kernelNode);

    hipKernelNodeParams kernelNodeParamsNew{};
    kernelNodeParamsNew.func = reinterpret_cast<void*>(addOneKernel);
    kernelNodeParamsNew.gridDim = dim3(1, 1, 1);
    kernelNodeParamsNew.blockDim = dim3(1, 1, 1);
    kernelNodeParamsNew.sharedMemBytes = 0;

    void* kernelArgsNew[2] = { reinterpret_cast<void*>(&devMemNew),
                            reinterpret_cast<void*>(&N) };
    kernelNodeParamsNew.kernelParams = kernelArgsNew;
    kernelNodeParamsNew.extra = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&kernelNodeNew, graph_out_org,
                                    nullptr, 0, &kernelNodeParamsNew));

    HIP_CHECK(dyn_hipStreamUpdateCaptureDependencies_ptr(stream,
              &kernelNodeNew, 1,
              hipStreamSetCaptureDependencies));

    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus_out_org,
                                         &id_out_org, &graph_out_org,
                                         &dependencies_out_org,
                                         &numDependencies_out_org));

    REQUIRE(numDependencies_out_org == 1);
    REQUIRE(dependencies_out_org[0] == kernelNodeNew);

    HIP_CHECK(hipMemcpyAsync(hostMem, devMemNew, Nbytes,
                             hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    #ifdef _WIN32
    HIP_CHECK(hipStreamSynchronize(stream));
    #endif

    REQUIRE(validateHostArray(hostMem, N, 2) == true);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  HIP_CHECK(hipFree(devMemNew));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphExecKernelNodeSetParams API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_hipGraphExecKernelNodeSetParams") {
  void* hipGraphExecKernelNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecKernelNodeSetParams",
            &hipGraphExecKernelNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphExecKernelNodeSetParams_ptr)(hipGraphExec_t,
    hipGraphNode_t, const hipKernelNodeParams *) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t,
    hipGraphNode_t, const hipKernelNodeParams *)>
    (hipGraphExecKernelNodeSetParams_ptr);

  int N = 1024;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 100);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);
  fillDeviceArray(devMem, N, 11);

  int* devMemNew = nullptr;
  HIP_CHECK(hipMalloc(&devMemNew, Nbytes));
  REQUIRE(devMemNew != nullptr);
  fillDeviceArray(devMemNew, N, 1);

  hipGraphNode_t kernelNode;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare kernelNode
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void* kernelParamArgs[2] = { reinterpret_cast<void*>(&devMem),
                               reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelParamArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
                                  nullptr, 0,
                                  &kernelNodeParams));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem, N, 12) == true);

  // Update graphExec with newKernelNodeParams in kernelNode
  hipKernelNodeParams newKernelNodeParams{};
  newKernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  newKernelNodeParams.gridDim = dim3(1, 1, 1);
  newKernelNodeParams.blockDim = dim3(1, 1, 1);
  newKernelNodeParams.sharedMemBytes = 0;

  void* kernelArgsNew[2] = { reinterpret_cast<void*>(&devMemNew),
                             reinterpret_cast<void*>(&N) };
  newKernelNodeParams.kernelParams = kernelArgsNew;
  newKernelNodeParams.extra = nullptr;

  HIP_CHECK(dyn_hipGraphExecKernelNodeSetParams_ptr(graphExec, kernelNode,
                                                    &newKernelNodeParams));

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMemNew, N, 2) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  HIP_CHECK(hipFree(devMemNew));

  free(hostMem);
}

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Graph related to User object from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_GraphUserObject") {
  void* hipGraphRetainUserObject_ptr = nullptr;
  void* hipGraphReleaseUserObject_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphRetainUserObject",
            &hipGraphRetainUserObject_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphReleaseUserObject",
            &hipGraphReleaseUserObject_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphRetainUserObject_ptr)(hipGraph_t, hipUserObject_t,
    unsigned int, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipGraph_t, hipUserObject_t,
    unsigned int, unsigned int)>
    (hipGraphRetainUserObject_ptr);
  hipError_t (*dyn_hipGraphReleaseUserObject_ptr)(hipGraph_t, hipUserObject_t,
    unsigned int) =
    reinterpret_cast<hipError_t (*)(hipGraph_t, hipUserObject_t,
    unsigned int)>
    (hipGraphReleaseUserObject_ptr);

  int* object = new int();
  REQUIRE(object != nullptr);

  int initialRefcount = 3;

  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObject,
                                initialRefcount,
                                hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("With_Count_1") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    HIP_CHECK(dyn_hipGraphRetainUserObject_ptr(graph, hObject, 1, 0));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));

    HIP_CHECK(hipUserObjectRelease(hObject, 3));
  }

  SECTION("With_Count_2") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    HIP_CHECK(dyn_hipGraphRetainUserObject_ptr(graph, hObject, 2, 0));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));

    HIP_CHECK(dyn_hipGraphReleaseUserObject_ptr(graph, hObject, 1));

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));

    HIP_CHECK(hipUserObjectRelease(hObject, 3));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipGraphDebugDotPrint API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_hipGraphDebugDotPrint") {
  void* hipGraphDebugDotPrint_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphDebugDotPrint",
            &hipGraphDebugDotPrint_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphDebugDotPrint_ptr)(hipGraph_t,
    const char *, unsigned int) =
    reinterpret_cast<hipError_t (*)(hipGraph_t,
    const char *, unsigned int)>
    (hipGraphDebugDotPrint_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 100);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraphNode_t memcpyNodeH2D, kernelNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNodeH2D, graph, nullptr, 0,
            devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

  ::std::vector<hipGraphNode_t> kernelNodeDependencies;
  kernelNodeDependencies.push_back(memcpyNodeH2D);

  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(addOneKernel);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(1, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;

  void* kernelArgs[2] = { reinterpret_cast<void*>(&devMem),
                          reinterpret_cast<void*>(&N) };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph,
            kernelNodeDependencies.data(), kernelNodeDependencies.size(),
            &kernelNodeParams));

  ::std::string fName("GraphData.dot");
  HIP_CHECK(dyn_hipGraphDebugDotPrint_ptr(graph, fName.c_str(),
                                          hipGraphDebugDotFlagsVerbose));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateDeviceArray(devMem, N, 101) == true);

  ::std::filesystem::path dir = ::std::filesystem::path(
                                ::std::filesystem::current_path().string());
  ::std::filesystem::path file = dir/fName;
  REQUIRE(::std::filesystem::exists(file) == true);
  REQUIRE(::std::filesystem::is_empty(file) == false);
  REQUIRE(remove(fName.c_str()) == 0);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipThreadExchangeStreamCaptureMode API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_ThreadExchangeStreamCaptureMode") {
  void* hipThreadExchangeStreamCaptureMode_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipThreadExchangeStreamCaptureMode",
            &hipThreadExchangeStreamCaptureMode_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipThreadExchangeStreamCaptureMode_ptr)(
    hipStreamCaptureMode *) =
    reinterpret_cast<hipError_t (*)(hipStreamCaptureMode *)>
    (hipThreadExchangeStreamCaptureMode_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode modeAtStart = GENERATE(
                                     hipStreamCaptureModeGlobal,
                                     hipStreamCaptureModeThreadLocal,
                                     hipStreamCaptureModeRelaxed);

  HIP_CHECK(hipStreamBeginCapture(stream, modeAtStart));

  hipStreamCaptureMode exchangeMode = GENERATE(
                                      hipStreamCaptureModeGlobal,
                                      hipStreamCaptureModeThreadLocal,
                                      hipStreamCaptureModeRelaxed);

  HIP_CHECK(dyn_hipThreadExchangeStreamCaptureMode_ptr(&exchangeMode));

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes,
                           hipMemcpyHostToDevice, stream));
  addOneKernel<<< 1, 1, 0, stream >>>(devMem , N);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(validateHostArray(hostMem, N, 11) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  free(hostMem);
  HIP_CHECK(hipFree(devMem));
}

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipDrvGraphMemcpyNodeSetParams
 *  - API, hipDrvGraphMemcpyNodeGetParams API from the hipGetProcAddress API
 *  - and then validates the basic functionality of those APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_hipDrvGraphMemcpySetGetParams") {
  CHECK_IMAGE_SUPPORT

  void* hipDrvGraphMemcpyNodeSetParams_ptr = nullptr;
  void* hipDrvGraphMemcpyNodeGetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipDrvGraphMemcpyNodeSetParams",
            &hipDrvGraphMemcpyNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipDrvGraphMemcpyNodeGetParams",
            &hipDrvGraphMemcpyNodeGetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipDrvGraphMemcpyNodeSetParams_ptr)(
    hipGraphNode_t, const HIP_MEMCPY3D *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, const HIP_MEMCPY3D *)>
    (hipDrvGraphMemcpyNodeSetParams_ptr);

  hipError_t (*dyn_hipDrvGraphMemcpyNodeGetParams_ptr)(
    hipGraphNode_t, HIP_MEMCPY3D *) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t, HIP_MEMCPY3D *)>
    (hipDrvGraphMemcpyNodeGetParams_ptr);

  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char *devMemSrc = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc),
                             &pitch, width, height));
  REQUIRE(devMemSrc != nullptr);

  char* hostMemDst1 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst1 != nullptr);
  fillCharHostArray(hostMemDst1, N, 100);

  char* hostMemDst2 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst2 != nullptr);
  fillCharHostArray(hostMemDst2, N, 100);

  hipGraphNode_t memsetNode, memcpyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Prepare memsetNode
  HIP_MEMSET_NODE_PARAMS memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  memsetParams.value = value;
  memsetParams.pitch = pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = width;
  memsetParams.height = height;

  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph,
                                             nullptr, 0,
                                             &memsetParams, context));

  // Prepare memcpyNode
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  HIP_MEMCPY3D memcpyParams{};
  memcpyParams.srcMemoryType = hipMemoryTypeDevice;
  memcpyParams.srcDevice = reinterpret_cast<hipDeviceptr_t>(devMemSrc);
  memcpyParams.srcPitch = pitch;
  memcpyParams.dstMemoryType = hipMemoryTypeHost;
  memcpyParams.dstHost = hostMemDst1;
  memcpyParams.dstPitch = width;
  memcpyParams.srcXInBytes = 0;
  memcpyParams.srcY = 0;
  memcpyParams.srcZ = 0;
  memcpyParams.dstXInBytes = 0;
  memcpyParams.dstY = 0;
  memcpyParams.dstZ = 0;
  memcpyParams.WidthInBytes = width;
  memcpyParams.Height = height;
  memcpyParams.Depth = 1;

  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph,
                                     memcpyNodeDependencies.data(),
                                     memcpyNodeDependencies.size(),
                                     &memcpyParams, context));

  HIP_MEMCPY3D receivedMemcpyValues{};
  HIP_CHECK(dyn_hipDrvGraphMemcpyNodeGetParams_ptr(memcpyNode,
                                              &receivedMemcpyValues));

  REQUIRE(receivedMemcpyValues.srcMemoryType == hipMemoryTypeDevice);
  REQUIRE(receivedMemcpyValues.srcDevice == devMemSrc);
  REQUIRE(receivedMemcpyValues.srcPitch == pitch);
  REQUIRE(receivedMemcpyValues.dstMemoryType == hipMemoryTypeHost);
  REQUIRE(receivedMemcpyValues.dstHost == hostMemDst1);
  REQUIRE(receivedMemcpyValues.dstPitch == width);
  REQUIRE(receivedMemcpyValues.srcXInBytes == 0);
  REQUIRE(receivedMemcpyValues.srcY == 0);
  REQUIRE(receivedMemcpyValues.srcZ == 0);
  REQUIRE(receivedMemcpyValues.dstXInBytes == 0);
  REQUIRE(receivedMemcpyValues.dstY == 0);
  REQUIRE(receivedMemcpyValues.dstZ == 0);
  REQUIRE(receivedMemcpyValues.WidthInBytes == width);
  REQUIRE(receivedMemcpyValues.Height == height);
  REQUIRE(receivedMemcpyValues.Depth == 1);

  HIP_MEMCPY3D correctedParms{};
  correctedParms.srcMemoryType = hipMemoryTypeDevice;
  correctedParms.srcDevice = reinterpret_cast<hipDeviceptr_t>(devMemSrc);
  correctedParms.srcPitch = pitch;
  correctedParms.dstMemoryType = hipMemoryTypeHost;
  correctedParms.dstHost = hostMemDst2;
  correctedParms.dstPitch = width;
  correctedParms.srcXInBytes = 0;
  correctedParms.srcY = 0;
  correctedParms.srcZ = 0;
  correctedParms.dstXInBytes = 0;
  correctedParms.dstY = 0;
  correctedParms.dstZ = 0;
  correctedParms.WidthInBytes = width;
  correctedParms.Height = height;
  correctedParms.Depth = 1;

  HIP_CHECK(dyn_hipDrvGraphMemcpyNodeSetParams_ptr(memcpyNode,
                                                   &correctedParms));

  HIP_CHECK(dyn_hipDrvGraphMemcpyNodeGetParams_ptr(memcpyNode,
                                                   &receivedMemcpyValues));

  REQUIRE(receivedMemcpyValues.srcMemoryType == hipMemoryTypeDevice);
  REQUIRE(receivedMemcpyValues.srcDevice == devMemSrc);
  REQUIRE(receivedMemcpyValues.srcPitch == pitch);
  REQUIRE(receivedMemcpyValues.dstMemoryType == hipMemoryTypeHost);
  REQUIRE(receivedMemcpyValues.dstHost == hostMemDst2);
  REQUIRE(receivedMemcpyValues.dstPitch == width);
  REQUIRE(receivedMemcpyValues.srcXInBytes == 0);
  REQUIRE(receivedMemcpyValues.srcY == 0);
  REQUIRE(receivedMemcpyValues.srcZ == 0);
  REQUIRE(receivedMemcpyValues.dstXInBytes == 0);
  REQUIRE(receivedMemcpyValues.dstY == 0);
  REQUIRE(receivedMemcpyValues.dstZ == 0);
  REQUIRE(receivedMemcpyValues.WidthInBytes == width);
  REQUIRE(receivedMemcpyValues.Height == height);
  REQUIRE(receivedMemcpyValues.Depth == 1);

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  #ifdef _WIN32
  HIP_CHECK(hipStreamSynchronize(0));
  #endif

  REQUIRE(validateArrayT<char>(hostMemDst1, N, 100) == true);
  REQUIRE(validateArrayT<char>(hostMemDst2, N, 120) == true);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMemSrc));
  free(hostMemDst1);
  free(hostMemDst2);

  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipGraphNodeSetParams API,
 *  - hipGraphExecNodeSetParams API from the hipGetProcAddress API
 *  - and then validates the basic functionality of those APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_GraphNodeSetParams") {
  CHECK_IMAGE_SUPPORT

  void* hipGraphNodeSetParams_ptr = nullptr;
  void* hipGraphExecNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipGraphNodeSetParams",
            &hipGraphNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipGraphExecNodeSetParams",
            &hipGraphExecNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGraphNodeSetParams_ptr)(
    hipGraphNode_t, hipGraphNodeParams*) =
    reinterpret_cast<hipError_t (*)(hipGraphNode_t,
                                    hipGraphNodeParams*)>
                                    (hipGraphNodeSetParams_ptr);
  hipError_t (*dyn_hipGraphExecNodeSetParams_ptr)(
    hipGraphExec_t, hipGraphNode_t, hipGraphNodeParams*) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                    hipGraphNodeParams*)>
                                    (hipGraphExecNodeSetParams_ptr);

  size_t width = 1024;
  size_t height = 1024;
  int N = width * height;
  int value = 120;
  size_t pitch;

  char *devMemSrc = nullptr;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc),
                             &pitch, width, height));
  REQUIRE(devMemSrc != nullptr);

  char* hostMemDst1 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst1 != nullptr);
  fillCharHostArray(hostMemDst1, N, 100);

  char* hostMemDst2 = reinterpret_cast<char *>(malloc( N * sizeof(char)));
  REQUIRE(hostMemDst2 != nullptr);
  fillCharHostArray(hostMemDst2, N, 100);

  hipGraphNode_t memsetNode, memcpyNode;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  memsetParams.value = value;
  memsetParams.pitch = pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = width;
  memsetParams.height = height;

  hipGraphNodeParams memsetNodeParams{};
  memsetNodeParams.type = hipGraphNodeTypeMemset;
  memsetNodeParams.memset = memsetParams;

  HIP_CHECK(hipGraphAddNode(&memsetNode, graph, nullptr, 0,
                            &memsetNodeParams));

  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  hipMemcpy3DParms initialParms{};
  initialParms.srcPos = make_hipPos(0, 0, 0);
  initialParms.dstPos = make_hipPos(0, 0, 0);
  initialParms.srcPtr = make_hipPitchedPtr(devMemSrc, pitch,
                                           width, height);
  initialParms.dstPtr = make_hipPitchedPtr(hostMemDst1, width,
                                           width, height);
  initialParms.extent = make_hipExtent(width, height, 1);
  initialParms.kind = hipMemcpyDeviceToHost;

  hipGraphNodeParams initialMemcpyNodeParams{};
  initialMemcpyNodeParams.type = hipGraphNodeTypeMemcpy;
  initialMemcpyNodeParams.memcpy.copyParams = initialParms;

  HIP_CHECK(hipGraphAddNode(&memcpyNode, graph, memcpyNodeDependencies.data(),
            memcpyNodeDependencies.size(), &initialMemcpyNodeParams));

  SECTION("Validate_hipGraphNodeSetParams") {
    hipMemcpy3DParms newParms{};
    newParms.srcPos = make_hipPos(0, 0, 0);
    newParms.dstPos = make_hipPos(0, 0, 0);
    newParms.srcPtr = make_hipPitchedPtr(devMemSrc, pitch, width, height);
    newParms.dstPtr = make_hipPitchedPtr(hostMemDst2, width,
                                         width, height);
    newParms.extent = make_hipExtent(width, height, 1);
    newParms.kind = hipMemcpyDeviceToHost;

    hipGraphNodeParams newMemcpyNodeParams{};
    newMemcpyNodeParams.type = hipGraphNodeTypeMemcpy;
    newMemcpyNodeParams.memcpy.copyParams = newParms;

    HIP_CHECK(dyn_hipGraphNodeSetParams_ptr(memcpyNode,
                                            &newMemcpyNodeParams));
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    #ifdef _WIN32
    HIP_CHECK(hipStreamSynchronize(0));
    #endif

    REQUIRE(validateArrayT<char>(hostMemDst1, N, 100) == true);
    REQUIRE(validateArrayT<char>(hostMemDst2, N, 120) == true);
    HIP_CHECK(hipGraphExecDestroy(graphExec));
  }

  SECTION("Validate_hipGraphExecNodeSetParams") {
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    #ifdef _WIN32
    HIP_CHECK(hipStreamSynchronize(0));
    #endif

    REQUIRE(validateArrayT<char>(hostMemDst1, N, 120) == true);
    REQUIRE(validateArrayT<char>(hostMemDst2, N, 100) == true);

    hipMemcpy3DParms newParms{};
    newParms.srcPos = make_hipPos(0, 0, 0);
    newParms.dstPos = make_hipPos(0, 0, 0);
    newParms.srcPtr = make_hipPitchedPtr(devMemSrc, pitch, width, height);
    newParms.dstPtr = make_hipPitchedPtr(hostMemDst2, width,
                                         width, height);
    newParms.extent = make_hipExtent(width, height, 1);
    newParms.kind = hipMemcpyDeviceToHost;

    hipGraphNodeParams newMemcpyNodeParams{};
    newMemcpyNodeParams.type = hipGraphNodeTypeMemcpy;
    newMemcpyNodeParams.memcpy.copyParams = newParms;

    HIP_CHECK(dyn_hipGraphExecNodeSetParams_ptr(graphExec, memcpyNode,
                                                &newMemcpyNodeParams));

    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    #ifdef _WIN32
    HIP_CHECK(hipStreamSynchronize(0));
    #endif

    REQUIRE(validateArrayT<char>(hostMemDst1, N, 120) == true);
    REQUIRE(validateArrayT<char>(hostMemDst2, N, 120) == true);
    HIP_CHECK(hipGraphExecDestroy(graphExec));
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(devMemSrc));
  free(hostMemDst1);
  free(hostMemDst2);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipDrvGraphExecMemsetNodeSetParams API from the hipGetProcAddress API
 *  - and then validates the basic functionality of that API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddressGraphApis.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_GraphAPIs_DrvGraphExecMemsetNodeSetParams") {
  CHECK_IMAGE_SUPPORT

  void* hipDrvGraphExecMemsetNodeSetParams_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipDrvGraphExecMemsetNodeSetParams",
            &hipDrvGraphExecMemsetNodeSetParams_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipDrvGraphExecMemsetNodeSetParams_ptr)(
    hipGraphExec_t, hipGraphNode_t,
    const HIP_MEMSET_NODE_PARAMS *, hipCtx_t) =
    reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipGraphNode_t,
                                    const HIP_MEMSET_NODE_PARAMS *, hipCtx_t)>
                                    (hipDrvGraphExecMemsetNodeSetParams_ptr);

  constexpr size_t width = 1024;
  constexpr size_t height = 1;
  constexpr int N = width * height;
  constexpr int value = 120;

  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  size_t pitch;
  hipDeviceptr_t devMemSrc;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void **>(&devMemSrc), &pitch,
                           width, height));

  char *hostMemDst = new char[N];
  REQUIRE(hostMemDst != nullptr);
  for (int i = 0; i < N; i++) {
    hostMemDst[i] = 0;
  }

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memsetNode, memcpyNode;

  // Prepare memset node
  HIP_MEMSET_NODE_PARAMS initialMemsetParams{};
  initialMemsetParams.dst = devMemSrc;
  initialMemsetParams.pitch = pitch;
  initialMemsetParams.elementSize = sizeof(char);
  initialMemsetParams.width = width;
  initialMemsetParams.height = height;
  initialMemsetParams.value = value;

  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                     &initialMemsetParams, context));

  // Prepare memcpyNode
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  HIP_MEMCPY3D memcpyParams{};
  memcpyParams.srcMemoryType = hipMemoryTypeDevice;
  memcpyParams.srcDevice = devMemSrc;
  memcpyParams.srcPitch = pitch;
  memcpyParams.dstMemoryType = hipMemoryTypeHost;
  memcpyParams.dstHost = hostMemDst;
  memcpyParams.dstPitch = width;
  memcpyParams.srcXInBytes = 0;
  memcpyParams.srcY = 0;
  memcpyParams.srcZ = 0;
  memcpyParams.dstXInBytes = 0;
  memcpyParams.dstY = 0;
  memcpyParams.dstZ = 0;
  memcpyParams.WidthInBytes = width;
  memcpyParams.Height = height;
  memcpyParams.Depth = 1;

  HIP_CHECK(hipDrvGraphAddMemcpyNode(
      &memcpyNode, graph, memcpyNodeDependencies.data(),
      memcpyNodeDependencies.size(), &memcpyParams, context));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, nullptr));
  HIP_CHECK(hipStreamSynchronize(0));

  for (int i = 0; i < N; i++) {
    REQUIRE(hostMemDst[i] == value);
  }

  HIP_MEMSET_NODE_PARAMS newMemsetParams{};
  newMemsetParams.dst = devMemSrc;
  newMemsetParams.pitch = pitch;
  newMemsetParams.elementSize = sizeof(char);
  newMemsetParams.width = width;
  newMemsetParams.height = height;
  newMemsetParams.value = value + 1;

  // Validating ipDrvGraphExecMemsetNodeSetParams API
  HIP_CHECK(dyn_hipDrvGraphExecMemsetNodeSetParams_ptr(graphExec,
            memsetNode, &newMemsetParams, context));

  for (int i = 0; i < N; i++) {
    hostMemDst[i] = 0;
  }

  HIP_CHECK(hipGraphLaunch(graphExec, nullptr));
  HIP_CHECK(hipStreamSynchronize(0));

  for (int i = 0; i < N; i++) {
    REQUIRE(hostMemDst[i] == (value + 1));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(reinterpret_cast<void *>(devMemSrc)));
  delete[] hostMemDst;
  HIP_CHECK(hipCtxDestroy(context));
}
