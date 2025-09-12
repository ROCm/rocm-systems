/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <utils.hh>
#include <resource_guards.hh>
#include <vector>
/*
This testcase verifies the following scenarios
1. H2H,H2PinMem and PinnedMem2Host
2. H2D-D2D-D2H in same GPU
*/
static constexpr auto NUM_ELM{1024 * 1024};
constexpr size_t N = 1000000;
constexpr unsigned blocks = 512;
constexpr unsigned threadsPerBlock = 256;
size_t Nbytes = N * sizeof(float);

TEST_CASE("Unit_hipMemcpyAsync_H2H-H2D-D2H-H2PinMem") {
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  int *A_Ph{nullptr}, *B_Ph{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h, &B_h, nullptr, NUM_ELM * sizeof(int));
  HipTest::initArrays<int>(nullptr, nullptr, nullptr, &A_Ph, &B_Ph, nullptr, NUM_ELM * sizeof(int),
                           true);

  SECTION("H2H, H2PinMem and PinMem2H") {
    HIP_CHECK(
        hipMemcpyWithStream(B_h, A_h, NUM_ELM * sizeof(int), hipMemcpyHostToHost, hipStreamLegacy));
    HIP_CHECK(hipMemcpyWithStream(A_Ph, B_h, NUM_ELM * sizeof(int), hipMemcpyHostToHost,
                                  hipStreamLegacy));
    HIP_CHECK(hipMemcpyWithStream(B_Ph, A_Ph, NUM_ELM * sizeof(int), hipMemcpyHostToHost,
                                  hipStreamLegacy));
    HipTest::checkTest(A_h, B_Ph, NUM_ELM);
  }

  SECTION("H2D-D2D-D2H-SameGPU") {
    HIP_CHECK(hipMemcpyWithStream(A_d, A_h, NUM_ELM * sizeof(int), hipMemcpyHostToDevice,
                                  hipStreamLegacy));
    HIP_CHECK(hipMemcpyWithStream(B_d, A_d, NUM_ELM * sizeof(int), hipMemcpyDeviceToDevice,
                                  hipStreamLegacy));
    HIP_CHECK(hipMemcpyWithStream(B_h, B_d, NUM_ELM * sizeof(int), hipMemcpyDeviceToHost,
                                  hipStreamLegacy));
    HipTest::checkTest(A_h, B_h, NUM_ELM);
  }
  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HipTest::freeArrays<int>(nullptr, nullptr, nullptr, A_Ph, B_Ph, nullptr, true);
}

TEST_CASE("Unit_hipStreamGetCaptureInfo_hipStreamLegacy_CaptureInfo") {
  hipStream_t stream{nullptr}, streamForGraph{nullptr};
  hipGraph_t graph{nullptr};
  hipError_t ret;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  float *A_d, *C_d;
  float *A_h, *C_h, *D_h;
  // Memory allocation to Host pointers
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  D_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);
  REQUIRE(D_h != nullptr);

  // Memory allocation to Device pointers
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = 1.0f + i;
    D_h[i] = 0.0f;
  }
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone},
      captureStatus1{hipStreamCaptureStatusNone}, captureStatus2{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID = 0,  // NOLINT
      capSequenceID1 = 0;

  // Verify the Error returned with null stream.
  SECTION("hipStreamGetCaptureInfo with hipStreamLegacy") {
    ret = hipStreamGetCaptureInfo(hipStreamLegacy, &captureStatus, &capSequenceID);
    REQUIRE(ret == hipErrorStreamCaptureImplicit);
  }
  SECTION("hipStreamGetCaptureInfo_v2 with hipStreamLegacy") {
    ret = hipStreamGetCaptureInfo_v2(hipStreamLegacy, &captureStatus, &capSequenceID, nullptr,
                                     nullptr, nullptr);
    REQUIRE(ret == hipErrorStreamCaptureImplicit);
  }


  // Check the capture status of the stream
  HIP_CHECK(hipStreamIsCapturing(stream, &captureStatus1));
  REQUIRE(captureStatus1 == hipStreamCaptureStatusActive);

  // Copy data to Device
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  // Kernal Operation
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  // End the capture
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  // Capture Status
  SECTION("hipStreamGetCaptureInfo with hipStreamLegacy after End capture") {
    ret = hipStreamGetCaptureInfo(hipStreamLegacy, &captureStatus2, &capSequenceID1);
    REQUIRE(ret == hipSuccess);
  }
  SECTION("hipStreamGetCaptureInfo_v2 with hipStreamLegacy after End capture") {
    ret = hipStreamGetCaptureInfo_v2(hipStreamLegacy, &captureStatus2, &capSequenceID1, nullptr,
                                     nullptr, nullptr);
    REQUIRE(ret == hipSuccess);
  }
  // Launch graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify Output
  for (size_t i = 0; i < N; i++) {
    D_h[i] = A_h[i] * A_h[i];
    REQUIRE(C_h[i] == D_h[i]);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
  free(D_h);
}

__global__ void MemPrefetchAsyncKernel(int* C_d, const int* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

TEST_CASE("Unit_hipMemPrefetchAsync_Basic") {
  LinearAllocGuard<int> alloc1(LinearAllocs::hipMallocManaged, kPageSize);
  const auto count = kPageSize / sizeof(*alloc1.ptr());
  constexpr auto fill_value = 42;
  std::fill_n(alloc1.ptr(), count, fill_value);


  HIP_CHECK(hipSetDevice(0));
  LinearAllocGuard<int> alloc2(LinearAllocs::hipMallocManaged, kPageSize);
  StreamGuard sg(Streams::created);
  HIP_CHECK(hipMemPrefetchAsync(alloc1.ptr(), kPageSize, 0, sg.stream()));
  MemPrefetchAsyncKernel<<<count / 1024 + 1, 1024, 0, sg.stream()>>>(alloc2.ptr(), alloc1.ptr(),
                                                                     count);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(sg.stream()));
  ArrayFindIfNot(alloc1.ptr(), fill_value, count);
  ArrayFindIfNot(alloc2.ptr(), fill_value * fill_value, count);

  HIP_CHECK(hipMemPrefetchAsync(alloc1.ptr(), kPageSize, hipCpuDeviceId, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(nullptr));
  ArrayFindIfNot(alloc1.ptr(), fill_value, count);
}

TEST_CASE("Unit_hipMemPoolApi_Basic") {
  int numElements = 64 * 1024 * 1024;
  float* A = nullptr;

  hipMemPool_t mem_pool = nullptr;
  int device = 0;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, device));
  HIP_CHECK(hipDeviceSetMemPool(device, mem_pool));
  HIP_CHECK(hipDeviceGetMemPool(&mem_pool, device));

  HIP_CHECK(
      hipMallocAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), hipStreamLegacy));
  INFO("hipMallocAsync result: " << A);

  HIP_CHECK(hipFreeAsync(A, hipStreamLegacy));
}
